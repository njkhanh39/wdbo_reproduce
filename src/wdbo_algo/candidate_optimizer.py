import numpy as np
import gpytorch
import torch
from wdbo_algo.candidate_model import learn_model_space_time
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
import wdbo_criterion
import math


def robust_scale(values, epsilon=1e-9):
	"""Return an outlier-resistant positive scale for acquisition values."""
	values = np.asarray(values, dtype=float).reshape(-1)
	if values.size == 0:
		return float(epsilon)
	q25, q75 = np.quantile(values, [0.25, 0.75])
	mad = np.median(np.abs(values - np.median(values))) * 1.4826
	return float(max(q75 - q25, mad, epsilon))


def gaussian_hellinger(mean_a, std_a, mean_b, std_b, epsilon=1e-12):
	"""Vectorized Hellinger distance between univariate Gaussian marginals."""
	mean_a = np.asarray(mean_a, dtype=float)
	mean_b = np.asarray(mean_b, dtype=float)
	std_a = np.maximum(np.asarray(std_a, dtype=float), epsilon)
	std_b = np.maximum(np.asarray(std_b, dtype=float), epsilon)
	denominator = std_a ** 2 + std_b ** 2
	affinity = np.sqrt(2.0 * std_a * std_b / denominator)
	affinity *= np.exp(-((mean_a - mean_b) ** 2) / (4.0 * denominator))
	return np.sqrt(np.clip(1.0 - affinity, 0.0, 1.0))


def jensen_shannon_choice_distance(values_a, values_b, temperature=1.0, epsilon=1e-12):
	"""Bounded JS metric between robustly scaled softmax choice vectors."""
	values_a = np.asarray(values_a, dtype=float).reshape(-1)
	values_b = np.asarray(values_b, dtype=float).reshape(-1)
	if values_a.shape != values_b.shape or values_a.size == 0:
		raise ValueError("choice vectors must be non-empty and have equal shape")
	if temperature <= 0:
		raise ValueError("temperature must be positive")
	shared_values = np.concatenate([values_a, values_b])
	shared_center = np.median(shared_values)
	scale = robust_scale(shared_values, epsilon)

	def probabilities(values):
		logits = (values - shared_center) / (temperature * scale)
		logits -= np.max(logits)
		weights = np.exp(logits)
		return weights / np.sum(weights)

	p = np.clip(probabilities(values_a), epsilon, 1.0)
	q = np.clip(probabilities(values_b), epsilon, 1.0)
	midpoint = 0.5 * (p + q)
	divergence = 0.5 * np.sum(p * np.log(p / midpoint))
	divergence += 0.5 * np.sum(q * np.log(q / midpoint))
	return float(np.sqrt(max(divergence, 0.0) / np.log(2.0)))


def dynamic_deletion_limit(dataset_size, maximum_fraction=0.25):
	"""Sublinear burst limit that grows with the current working-set size."""
	if dataset_size < 1:
		raise ValueError("dataset_size must be positive")
	if not 0 < maximum_fraction <= 1:
		raise ValueError("maximum_fraction must lie in (0, 1]")
	return max(1, min(math.ceil(math.sqrt(dataset_size)), math.ceil(maximum_fraction * dataset_size)))


def joint_gate_load(nas, normalized_margin, hellinger, hellinger_threshold, epsilon=1e-12):
	"""Return maximum normalized utilization of acquisition and posterior gates."""
	acquisition_load = 2.0 * np.asarray(nas, dtype=float) / max(float(normalized_margin), epsilon)
	posterior_load = np.asarray(hellinger, dtype=float) / float(hellinger_threshold)
	return np.maximum(acquisition_load, posterior_load)


def effective_dimension(kernel_eigenvalues, noise, epsilon=1e-12):
	"""Effective GP dimension tr(K(K + noise I)^-1)."""
	eigenvalues = np.maximum(np.asarray(kernel_eigenvalues, dtype=float), 0.0)
	return float(np.sum(eigenvalues / (eigenvalues + max(float(noise), epsilon))))


def single_prune_cost(gate_load, base_cost=0.1):
	"""Risk-adjusted credit required for one deletion."""
	if not 0 < base_cost <= 1:
		raise ValueError("base_cost must lie in (0, 1]")
	return float(base_cost + (1.0 - base_cost) * np.clip(gate_load, 0.0, 1.0))


def update_single_prune_credit(credit, elapsed_time, temporal_lengthscale, alpha,
		data_size, reserve_size, credit_cap=1.0, epsilon=1e-12):
	"""Refill a bounded token bucket using drift and dataset pressure."""
	if credit_cap <= 0:
		raise ValueError("credit_cap must be positive")
	pressure = np.clip((data_size - reserve_size) / max(reserve_size, 1), 0.0, 1.0)
	drift = max(float(elapsed_time), 0.0) / max(float(temporal_lengthscale), epsilon)
	refill = float(alpha) * drift * (1.0 + pressure)
	return float(min(credit_cap, max(0.0, credit) + refill)), float(pressure), refill

class CandidatePruningOptimizer:
	"""WDBO Optimizer class, interfaces with the user
	"""

	def __init__(
			self,
			spatial_domain,
			spatial_kernel,
			temporal_kernel,
			spatial_kernel_args=None,
			temporal_kernel_args=None,
			n_initial_observations=15,
			min_dataset_size=2,
			alpha=0.25,
			pruning_method="wasserstein",
			candidate_pool_size=128,
			contender_kappa=1.0,
			hellinger_threshold=0.1,
			js_temperature=1.0,
			max_deletions_per_clean=2,
			max_deletion_fraction=0.25,
			single_budget_base_cost=0.1,
			single_budget_min_reserve=10,
			single_budget_effective_dim_multiplier=1.25,
			single_budget_credit_cap=1.0,
	):
		"""Build the WDBO algorithm

		Args:
				spatial_domain (np.array): `d x 2`-array describing a `d`-dimensional hyperrectangle
				spatial_kernel (gpytorch.kernels.Kernel class): the spatial kernel class
				temporal_kernel (gpytorch.kernels.Kernel class): the temporal kernel class
				spatial_kernel_args (list, optional): the arguments for building the spatial kernel. Defaults to [].
				temporal_kernel_args (list, optional): the arguments for building the temporal kernel. Defaults to [].
				n_initial_observations (int, optional): the number of observations to collect before starting the optimization.
				Defaults to 15.
				alpha (float, optional): the WDBO hyperparameter, control the removal budget. Defaults to 0.25.
		"""
		self._spatial_domain = spatial_domain
		self._d = self._spatial_domain.shape[0]

		self._n_initial_observations = n_initial_observations
		if min_dataset_size < 2:
			raise ValueError("min_dataset_size must be at least 2")
		self._min_dataset_size = int(min_dataset_size)
		self._alpha = alpha

		self._spatial_kernel = spatial_kernel
		self._spatial_kernel_wdbo = self.get_wdbo_kernel_class(self._spatial_kernel)
		self._spatial_kernel_args = [] if spatial_kernel_args is None else list(spatial_kernel_args)

		self._temporal_kernel = temporal_kernel
		self._temporal_kernel_wdbo = self.get_wdbo_kernel_class(self._temporal_kernel)
		self._temporal_kernel_args = [] if temporal_kernel_args is None else list(temporal_kernel_args)

		allowed_methods = {"wasserstein", "nas", "hellinger", "js", "dual_gate", "dual_gate_budget", "joint_dual_gate"}
		if pruning_method not in allowed_methods:
			raise ValueError(f"pruning_method must be one of {sorted(allowed_methods)}")
		if candidate_pool_size < 8:
			raise ValueError("candidate_pool_size must be at least 8")
		if contender_kappa <= 0:
			raise ValueError("contender_kappa must be positive")
		if not 0 < hellinger_threshold <= 1:
			raise ValueError("hellinger_threshold must lie in (0, 1]")
		if js_temperature <= 0:
			raise ValueError("js_temperature must be positive")
		if max_deletions_per_clean is not None and max_deletions_per_clean < 1:
			raise ValueError("max_deletions_per_clean must be positive or None")
		if not 0 < max_deletion_fraction <= 1:
			raise ValueError("max_deletion_fraction must lie in (0, 1]")
		if not 0 < single_budget_base_cost <= 1:
			raise ValueError("single_budget_base_cost must lie in (0, 1]")
		if single_budget_min_reserve < 2:
			raise ValueError("single_budget_min_reserve must be at least 2")
		if single_budget_effective_dim_multiplier <= 0 or single_budget_credit_cap <= 0:
			raise ValueError("single budget multiplier and cap must be positive")
		self._pruning_method = pruning_method
		self._candidate_pool_size = int(candidate_pool_size)
		self._contender_kappa = float(contender_kappa)
		self._hellinger_threshold = float(hellinger_threshold)
		self._js_temperature = float(js_temperature)
		self._max_deletions_per_clean = max_deletions_per_clean
		self._max_deletion_fraction = float(max_deletion_fraction)
		self._single_budget_base_cost = float(single_budget_base_cost)
		self._single_budget_min_reserve = int(single_budget_min_reserve)
		self._single_budget_effective_dim_multiplier = float(single_budget_effective_dim_multiplier)
		self._single_budget_credit_cap = float(single_budget_credit_cap)
		self._single_prune_credit = 0.0
		self.last_clean_diagnostics = {}

		self._xx_tt = None
		self._yy = None
		self._budget = 1.0

		self._gpr = None
		self._lambda, self._lS, self._lT, self._noise = None, None, None, None
		self._current_time = None

	def get_wdbo_kernel_class(self, gpytorch_kernel_class):
		"""Correspondance between gpytorch kernels classes and wdbo-criterion kernels classes.

		Args:
				gpytorch_kernel_class (gpytorch.kernels.Kernel class): the kernel class in gpytorch

		Returns:
				wdbo_criterion.Kernel class: the kernel class in wdbo_criterion
		"""
		if gpytorch_kernel_class == gpytorch.kernels.RBFKernel:
			return wdbo_criterion.RBFKernel
		if gpytorch_kernel_class == gpytorch.kernels.MaternKernel:
			return wdbo_criterion.MaternKernel

		return None

	def dataset_size(self):
		"""Compute the dataset size of the DBO algorithm

		Returns:
				int: the dataset size
		"""
		return self._xx_tt.shape[0]

	def denormalize_x(self, x):
		"""Linear map from [0, 1]^d to the spatial domain of the objective function

		Args:
				x (np.array): the input

		Returns:
				np.array: the input mapped in the function domain
		"""
		return x * (self._spatial_domain[:, 1] - self._spatial_domain[:, 0]) + self._spatial_domain[:, 0]

	def normalize_x(self, x):
		"""Linear map from the spatial domain of the objective function to [0, 1]^d

		Args:
				x (np.array): the input

		Returns:
				np.array: the input mapped in [0, 1]^d
		"""
		return (x - self._spatial_domain[:, 0]) / (self._spatial_domain[:, 1] - self._spatial_domain[:, 0])

	def normalize_y(self, y):
		"""Standardize the input (i.e. subtract the empirical mean, divide by the empirical standard deviation)

		Args:
				y (np.array): the input

		Returns:
				np.array: the input standardized
		"""
		scale = float(np.std(y))
		if not np.isfinite(scale) or scale < 1e-12:
			scale = 1.0
		return (y - np.mean(y)) / scale

	def _predictive_stats(self, model, points):
		"""Return predictive mean/std at a finite candidate set."""
		points_tensor = torch.as_tensor(points, dtype=torch.float64)
		with torch.no_grad(), gpytorch.settings.fast_pred_var():
			posterior = model.posterior(points_tensor)
			mean = posterior.mean.detach().cpu().numpy().reshape(-1)
			std = posterior.variance.clamp_min(1e-12).sqrt().detach().cpu().numpy().reshape(-1)
		return mean, std

	def _leave_one_out_predictive_stats(self, points, full_mean, full_std):
		"""Return every fixed-hyperparameter LOO posterior in one factorization.

		For an exact GP, deleting observation ``i`` changes the predictive mean
		and variance through one column of ``(K + noise I)^-1``.  Using this
		identity is algebraically equivalent to constructing ``n`` reduced GPs,
		but avoids ``n`` model objects and ``n`` repeated factorizations at every
		cleaning step.
		"""
		train_x = torch.as_tensor(self._xx_tt, dtype=torch.float64)
		train_y = torch.as_tensor(self._yy_normalized, dtype=torch.float64)
		test_x = torch.as_tensor(points, dtype=torch.float64)
		with torch.no_grad():
			train_mean = self._gpr.mean_module(train_x)
			kernel = self._gpr.covar_module(train_x).to_dense()
			noise = self._gpr.likelihood.noise.detach().reshape(())
			observation_covariance = kernel + noise * torch.eye(
				len(train_x), dtype=torch.float64, device=train_x.device
			)
			cholesky = torch.linalg.cholesky(observation_covariance)
			centered_y = (train_y - train_mean).unsqueeze(-1)
			alpha = torch.cholesky_solve(centered_y, cholesky).squeeze(-1)
			inverse = torch.cholesky_inverse(cholesky)
			cross_covariance = self._gpr.covar_module(test_x, train_x).to_dense()
			cross_times_inverse = cross_covariance @ inverse
			inverse_diagonal = torch.diagonal(inverse).clamp_min(1e-15)

			mean_shift = cross_times_inverse * (alpha / inverse_diagonal).unsqueeze(0)
			variance_increase = cross_times_inverse.square() / inverse_diagonal.unsqueeze(0)
			loo_mean = torch.as_tensor(full_mean, dtype=torch.float64).unsqueeze(1) - mean_shift
			loo_variance = torch.as_tensor(full_std, dtype=torch.float64).square().unsqueeze(1)
			loo_variance = loo_variance + variance_increase

		return (
			loo_mean.cpu().numpy(),
			loo_variance.clamp_min(1e-12).sqrt().cpu().numpy(),
		)

	def _candidate_reference(self, current_time):
		"""Freeze the decision boundary at the start of one cleaning transaction."""
		seed = int(round(float(current_time) * 1_000_000)) + 104729
		engine = torch.quasirandom.SobolEngine(self._d, scramble=True, seed=seed)
		space = engine.draw(self._candidate_pool_size).double().numpy()
		points = np.column_stack([space, np.full(self._candidate_pool_size, current_time)])
		full_mean, full_std = self._predictive_stats(self._gpr, points)
		beta = 0.2 * self._d * np.log(2 * self._xx_tt.shape[0])
		full_acquisition = full_mean + np.sqrt(max(beta, 1e-12)) * full_std
		scale = robust_scale(full_acquisition)
		order = np.argsort(full_acquisition)[::-1]
		margin = float(full_acquisition[order[0]] - full_acquisition[order[1]])
		normalized_margin = margin / scale
		contenders = (full_acquisition.max() - full_acquisition) <= self._contender_kappa * scale
		if np.count_nonzero(contenders) < 2:
			contenders[order[:min(8, len(order))]] = True
		return {
			"points": points,
			"mean": full_mean,
			"std": full_std,
			"acquisition": full_acquisition,
			"beta": beta,
			"scale": scale,
			"normalized_margin": normalized_margin,
			"contenders": contenders,
		}

	def _candidate_relevance(self, current_time, reference=None):
		"""Compute finite-pool leave-one-out scores for candidate pruning methods."""
		# Joint Dual Gate compares every proposed post-deletion model with the model
		# before the whole clean call. This measures accumulated, not merely local,
		# perturbation and keeps every accepted deletion under one certificate.
		if reference is None:
			local_reference = self._candidate_reference(current_time)
			points = local_reference["points"]
			full_mean = local_reference["mean"]
			full_std = local_reference["std"]
			loo_mean, loo_std = self._leave_one_out_predictive_stats(points, full_mean, full_std)
			beta = local_reference["beta"]
			full_acquisition = local_reference["acquisition"]
			scale = local_reference["scale"]
			normalized_margin = local_reference["normalized_margin"]
			contenders = local_reference["contenders"]
		else:
			points = reference["points"]
			current_mean, current_std = self._predictive_stats(self._gpr, points)
			loo_mean, loo_std = self._leave_one_out_predictive_stats(points, current_mean, current_std)
			full_mean = reference["mean"]
			full_std = reference["std"]
			beta = reference["beta"]
			full_acquisition = reference["acquisition"]
			scale = reference["scale"]
			normalized_margin = reference["normalized_margin"]
			contenders = reference["contenders"]

		nas = np.empty(len(self._yy), dtype=float)
		hellinger = np.empty(len(self._yy), dtype=float)
		js = np.empty(len(self._yy), dtype=float)
		for index in range(len(self._yy)):
			loo_acquisition = loo_mean[:, index] + np.sqrt(max(beta, 1e-12)) * loo_std[:, index]
			nas[index] = np.max(np.abs(full_acquisition[contenders] - loo_acquisition[contenders])) / scale
			hellinger[index] = np.max(gaussian_hellinger(
				full_mean[contenders], full_std[contenders],
				loo_mean[contenders, index], loo_std[contenders, index],
			))
			js[index] = jensen_shannon_choice_distance(
				full_acquisition[contenders], loo_acquisition[contenders], self._js_temperature
			)

		if self._pruning_method == "nas":
			relevance = nas
			eligible = np.ones(len(nas), dtype=bool)
		elif self._pruning_method == "hellinger":
			relevance = hellinger
			eligible = np.ones(len(nas), dtype=bool)
		elif self._pruning_method == "js":
			relevance = js
			eligible = np.ones(len(nas), dtype=bool)
		elif self._pruning_method == "dual_gate":
			relevance = nas
			eligible = (2.0 * nas < normalized_margin) & (hellinger < self._hellinger_threshold)
		else:
			relevance = joint_gate_load(nas, normalized_margin, hellinger, self._hellinger_threshold)
			eligible = relevance < 1.0

		return relevance, eligible, {
			"normalized_margin": normalized_margin,
			"acquisition_scale": scale,
			"contender_count": int(np.count_nonzero(contenders)),
			"nas_min": float(np.min(nas)),
			"hellinger_min": float(np.min(hellinger)),
			"js_min": float(np.min(js)),
			"joint_load_min": float(np.min(relevance)) if self._pruning_method == "joint_dual_gate" else "",
		}

	def _rebuild_surrogate_with_fixed_hyperparameters(self):
		"""Condition on the reduced set without fitting or renormalizing again."""
		state = self._gpr.state_dict()
		model = learn_model_space_time(
			self._xx_tt,
			self._spatial_kernel,
			self._spatial_kernel_args,
			self._temporal_kernel,
			self._temporal_kernel_args,
			self._yy_normalized,
			fit_model=False,
		).double()
		model.load_state_dict(state)
		model.eval()
		model.likelihood.eval()
		self._gpr = model

	def _effective_dimension_and_reserve(self):
		train_x = torch.as_tensor(self._xx_tt, dtype=torch.float64)
		with torch.no_grad():
			kernel = self._gpr.covar_module(train_x).to_dense()
			eigenvalues = torch.linalg.eigvalsh(kernel).clamp_min(0.0).cpu().numpy()
		dimension = effective_dimension(eigenvalues, self._noise)
		reserve = max(
			self._min_dataset_size,
			self._single_budget_min_reserve,
			math.ceil(self._single_budget_effective_dim_multiplier * dimension),
		)
		return dimension, min(reserve, self._xx_tt.shape[0])

	def _clean_sequential_with_dynamic_budget(self, t, verbose=False):
		"""Prune sequentially, paying and recomputing after every deletion."""
		dimension, reserve = self._effective_dimension_and_reserve()
		elapsed = t - self._current_time
		self._single_prune_credit, pressure, refill = update_single_prune_credit(
			self._single_prune_credit,
			elapsed,
			self._lT,
			self._alpha,
			self._xx_tt.shape[0],
			reserve,
			self._single_budget_credit_cap,
		)
		deleted = 0
		spent = 0.0
		self.last_clean_diagnostics = {
			"method": self._pruning_method,
			"deleted": deleted,
			"effective_dimension": dimension,
			"dataset_reserve": reserve,
			"single_prune_credit": self._single_prune_credit,
			"budget_pressure": pressure,
			"budget_refill": refill,
			"sequential_spend": spent,
		}

		while True:
			dimension, reserve = self._effective_dimension_and_reserve()
			self.last_clean_diagnostics.update({
				"effective_dimension": dimension,
				"dataset_reserve": reserve,
			})
			if self._xx_tt.shape[0] <= reserve:
				self.last_clean_diagnostics["stopped"] = "dataset_reserve"
				break

			# Each score is local to the current dataset. If a point is removed,
			# refit and recompute before considering the next point. This is
			# sequential single-point accounting, not joint subset accounting.
			gate_load, eligible, candidate_diagnostics = self._candidate_relevance(t)
			self.last_clean_diagnostics.update(candidate_diagnostics)
			if not np.any(eligible):
				self.last_clean_diagnostics["stopped"] = "no_candidate_passed_gate"
				break

			masked_load = np.where(eligible, gate_load, np.inf)
			idx_min = int(np.argmin(masked_load))
			cost = single_prune_cost(masked_load[idx_min], self._single_budget_base_cost)
			self.last_clean_diagnostics.update({
				"gate_load_min": float(masked_load[idx_min]),
				"single_prune_cost": cost,
			})
			if self._single_prune_credit < cost:
				self.last_clean_diagnostics["stopped"] = "insufficient_credit"
				break

			self._single_prune_credit -= cost
			spent += cost
			self._xx_tt = np.delete(self._xx_tt, idx_min, axis=0)
			self._yy = np.delete(self._yy, idx_min, axis=0)
			self.update_surrogate_model(verbose=verbose)
			deleted += 1
			self.last_clean_diagnostics.update({
				"deleted": deleted,
				"last_index": idx_min,
				"single_prune_credit": self._single_prune_credit,
				"sequential_spend": spent,
			})

	def update_surrogate_model(self, verbose=False):
		"""Conditon a Gaussian Process on the collected data

		Args:
				verbose (bool, optional): verbose output. Defaults to False.
		"""
		self._yy_normalized = self.normalize_y(self._yy)
		self._gpr = learn_model_space_time(self._xx_tt, self._spatial_kernel, self._spatial_kernel_args, self._temporal_kernel, self._temporal_kernel_args, self._yy_normalized)
		self._lambda, self._lS, self._lT, self._noise = np.exp(self._gpr.get_kernel_log_hyperparameters())
		if verbose:
			print(f"Hyperparameters (lambda, lS, lT, noise variance) = {(self._lambda, self._lS, self._lT, self._noise)}")

	def tell(self, x, t, y, verbose=False):
		"""Add an observation to the dataset and update the surrogate model.

		Args:
				x (np.array): the input in the space domain
				t (float): the input in the time domain
				y (float): the noisy output
				verbose (bool, optional): verbose output. Defaults to False.
		"""
		normalized_x = self.normalize_x(x)

		# Add the input output pair to the dataset
		if self._xx_tt is None:
			self._xx_tt = np.array([np.concatenate((normalized_x, np.array([t])))])
			self._yy = np.array([y])
		else:
			self._xx_tt = np.concatenate((self._xx_tt, np.array([np.concatenate((normalized_x, np.array([t])))])))
			self._yy = np.concatenate((self._yy, np.array([y])))

		if self._n_initial_observations > 0:
			self._n_initial_observations -= 1

		# Update the surrogate model and the hyperparameters
		if self._n_initial_observations <= 0:
			self.update_surrogate_model(verbose=verbose)

	def next_query(self, current_time):
		"""Find the next relevant input to query.

		Args:
				current_time (float): the present time

		Returns:
				np.array: a relevant input to query in the space domain
		"""
		# Initial observations, without optimization of the acquisition function
		if self._n_initial_observations > 0:
			random_normalized_x = np.random.uniform(low=0.0, high=1.0, size=(self._d,))
			return self.denormalize_x(random_normalized_x)

		# The initial observations are gathered, we now have to optimize the acquisition function
		UCB = UpperConfidenceBound(self._gpr, beta=0.2 * self._d * np.log(2 * self._xx_tt.shape[0]))
		low_bounds = torch.zeros(self._d+1, dtype=torch.float64)
		low_bounds[-1] = current_time
		up_bounds = torch.ones(self._d+1, dtype=torch.float64)
		up_bounds[-1] = current_time
		bounds = torch.stack([low_bounds, up_bounds])
		candidate, _ = optimize_acqf(UCB, bounds=bounds, q=1, num_restarts=20, raw_samples=512,)
		next_sample = candidate[0].numpy()

		return self.denormalize_x(next_sample[:-1])

	def clean(self, t, verbose=False):
		"""Remove irrelevant observations from the dataset

		Args:
				t (float): the present time
				verbose (bool, optional): verbose output. Defaults to False.
		"""
		if self._n_initial_observations > 0:
			return

		if self._current_time is None:
			self._current_time = t

		if self._pruning_method == "dual_gate_budget":
			self._clean_sequential_with_dynamic_budget(t, verbose=verbose)
			self._current_time = t
			return

		self._budget *= (1.0 + self._alpha) ** ((t - self._current_time) / self._lT)

		# Cleaning loop for the dataset
		min_crit = 0
		deleted = 0
		joint_load_committed = 0.0
		joint_reference = self._candidate_reference(t) if self._pruning_method == "joint_dual_gate" else None
		if self._pruning_method == "joint_dual_gate" and self._max_deletions_per_clean is None:
			deletion_limit = dynamic_deletion_limit(self._xx_tt.shape[0], self._max_deletion_fraction)
		else:
			deletion_limit = self._max_deletions_per_clean
		self.last_clean_diagnostics = {
			"method": self._pruning_method,
			"deleted": 0,
			"deletion_limit": "" if deletion_limit is None else deletion_limit,
		}
		while self._xx_tt.shape[0] > self._min_dataset_size and self._budget > min_crit:
			if deletion_limit is not None and deleted >= deletion_limit:
				self.last_clean_diagnostics["stopped"] = "deletion_limit"
				break
			# Measures observations relevancy
			if self._pruning_method == "wasserstein":
				criteria = wdbo_criterion.wasserstein_criterion(
					np.ascontiguousarray(self._xx_tt[:, :-1]),
					np.ascontiguousarray(self._yy_normalized),
					np.ascontiguousarray(self._xx_tt[:, -1]),
					self._xx_tt.shape[0],
					self._d,
					self._lambda, self._noise,
					self._spatial_kernel_wdbo(*([self._lS] + self._spatial_kernel_args)), self._temporal_kernel_wdbo(*([self._lT] + self._temporal_kernel_args)),
					t,
					0, 1)
				eligible = np.ones(len(criteria), dtype=bool)
				candidate_diagnostics = {}
			else:
				criteria, eligible, candidate_diagnostics = self._candidate_relevance(t, reference=joint_reference)
				self.last_clean_diagnostics.update(candidate_diagnostics)
				joint_total_criteria = criteria.copy() if self._pruning_method == "joint_dual_gate" else None
				if joint_total_criteria is not None:
					# The global removal budget pays only the increase in the joint
					# perturbation envelope. Paying the cumulative load again after
					# every deletion would systematically under-clean large sets.
					criteria = np.maximum(0.0, joint_total_criteria - joint_load_committed)

			if not np.any(eligible):
				self.last_clean_diagnostics["stopped"] = "no_candidate_passed_gate"
				break

			# Find the least relevant observation
			masked_criteria = np.where(eligible, criteria, np.inf)
			sorted_args = masked_criteria.argsort()
			indices, criteria = sorted_args, criteria[sorted_args]
			idx_min, min_crit = (indices[0], criteria[0] + 1.0)

			if verbose:
				print(f"Removal Budget: {self._budget} // Least Relevant Observation: {idx_min} // Relevancy: {min_crit} (i.e. {round(100 * min_crit / self._budget, 2)}% of budget)")

			# Remove it if the budget allows it
			if min_crit < self._budget:
				# Budget consumption
				if min_crit > 1.0:
					self._budget = self._budget / min_crit
				if self._pruning_method == "joint_dual_gate":
					joint_load_committed = float(joint_total_criteria[idx_min])

				if verbose:
					print(f"Observation {idx_min} is removed")

				# Dataset update
				self._xx_tt = np.delete(self._xx_tt, (idx_min), axis=0)
				self._yy = np.delete(self._yy, (idx_min), axis=0)
				if self._pruning_method == "joint_dual_gate":
					self._yy_normalized = np.delete(self._yy_normalized, (idx_min), axis=0)
					self._rebuild_surrogate_with_fixed_hyperparameters()
				else:
					self.update_surrogate_model(verbose=verbose)
				deleted += 1
				self.last_clean_diagnostics.update({
					"deleted": deleted,
					"last_index": idx_min,
					"last_cost": min_crit,
					"joint_load_committed": joint_load_committed if self._pruning_method == "joint_dual_gate" else "",
				})

		self._current_time = t
