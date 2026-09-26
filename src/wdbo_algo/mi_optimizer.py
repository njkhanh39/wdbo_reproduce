import numpy as np
import gpytorch
from wdbo_algo.mi_criterion import mi_criterion
from wdbo_algo.optimizer import WDBOOptimizer

class MIDBOOptimizer(WDBOOptimizer):
	"""W-DBO with the Wasserstein removal criterion replaced by a mutual-information one.

	Only `clean()` differs. The surrogate model, the hyperparameter estimation, the
	UCB acquisition, the normalization and the data structures are all inherited
	unchanged, so a run of this class and a run of `WDBOOptimizer` differ in exactly
	one thing: which observations get deleted, and when.

	Two differences in the removal rule itself, both forced by the change of units.
	W-DBO scores an observation by a Wasserstein distance normalized against the
	prior, so its scores are dimensionless and bounded, and its budget is
	multiplicative: `b *= (1 + alpha) ** (dt / lT)`, consumed by division.
	`mi_criterion` returns nats, which are additive, so:

	- the budget accrues additively, `b += alpha * dt / lT` -- alpha nats of
	  discardable information per temporal lengthscale of elapsed time;
	- it is consumed by subtraction, `b -= R(i)`;
	- it starts at 0.0 rather than 1.0. Nothing should be removable before any time
	  has passed, and there is no `+1` offset to undo.

	Taking logs of W-DBO's own loop shows the two rules are the same algebra:
	`log b += log(1 + alpha) . dt / lT` and `log b -= log(1 + criterion)`. W-DBO's
	alpha = 1/4 is therefore "discard 0.223 nats per lengthscale". It works at that
	scale because its criterion is normalized into [0, 1]; `mi_criterion` is not
	normalized, and its scores on real datasets run from 1e-24 to 1e-3 nats, so
	alpha here is many orders of magnitude smaller and is not transferable between
	the two rules -- nor, in general, between temporal lengthscales.
	"""

	def __init__(self, spatial_domain, spatial_kernel, temporal_kernel, spatial_kernel_args=[],
	             temporal_kernel_args=[], n_initial_observations=15, alpha=1e-11,
	             min_dataset_size=15, budget_cap=None, max_removals_per_clean=None,
	             n_times=8, horizon_lengthscales=3.0, n_max_samples=32, n_candidates=512,
	             weight="kernel", fstar_source="loo", clip_horizon=None, seed=None):
		"""Build the MI-criterion DBO algorithm.

		Args:
				spatial_domain (np.array): `d x 2`-array describing a `d`-dimensional hyperrectangle
				spatial_kernel (gpytorch.kernels.Kernel class): the spatial kernel class; must be MaternKernel
				temporal_kernel (gpytorch.kernels.Kernel class): the temporal kernel class; must be MaternKernel
				spatial_kernel_args (list, optional): the arguments for building the spatial kernel. Defaults to [].
				temporal_kernel_args (list, optional): the arguments for building the temporal kernel. Defaults to [].
				n_initial_observations (int, optional): observations to collect before optimizing. Defaults to 15.
				alpha (float, optional): nats of discardable information accrued per temporal lengthscale.
				Defaults to 1e-11, measured on ackley4d -- see experiments/MI_CRITERION.md.
				min_dataset_size (int, optional): never remove below this many observations. Defaults to 15,
				matching the initial design, because the criterion collapses towards zero for every
				observation at once when the estimated lT is small.
				budget_cap (float, optional): clamp the accrued budget to this many nats, so a clean that
				removes nothing cannot bank budget and purge a burst later. Defaults to None (uncapped).
				max_removals_per_clean (int, optional): stop after this many removals in one call, regardless
				of budget. Each removal triggers a full hyperparameter refit, so this bounds the worst-case
				response time. Defaults to None (bounded only by `min_dataset_size`).
				n_times (int, optional): quadrature nodes over future time. Defaults to 8.
				horizon_lengthscales (float, optional): how far ahead to integrate. Defaults to 3.0.
				n_max_samples (int, optional): Monte-Carlo samples of f*_t. Defaults to 32.
				n_candidates (int, optional): candidate points discretizing the space. Defaults to 512.
				weight (str, optional): "kernel" or "uniform" weighting of future times. Defaults to "kernel".
				fstar_source (str, optional): "loo" samples f*_t from each leave-one-out posterior, "full"
				from the full-data posterior once per future time, "is" reweights those full-data samples
				towards each leave-one-out posterior by importance sampling. Defaults to "loo".
				clip_horizon (float, optional): cap the future horizon at this absolute time. Defaults to None.
				seed (int, optional): seed for the criterion's own sampling. Defaults to None.
		"""
		super().__init__(spatial_domain, spatial_kernel, temporal_kernel,
		                 spatial_kernel_args=spatial_kernel_args,
		                 temporal_kernel_args=temporal_kernel_args,
		                 n_initial_observations=n_initial_observations,
		                 alpha=alpha, min_dataset_size=min_dataset_size)

		for name, kernel_class in (("spatial", spatial_kernel), ("temporal", temporal_kernel)):
			if kernel_class is not gpytorch.kernels.MaternKernel:
				raise ValueError(f"mi_criterion only implements Matern kernels, got {name} {kernel_class.__name__}")

		# MaternKernel takes nu as its first positional argument; the surrogate model
		# is built with the same list, so reading it here keeps the two in step.
		self._spatial_nu = spatial_kernel_args[0] if spatial_kernel_args else 2.5
		self._temporal_nu = temporal_kernel_args[0] if temporal_kernel_args else 2.5

		self._budget = 0.0
		self._budget_cap = budget_cap
		self._max_removals_per_clean = max_removals_per_clean
		self._rng = np.random.default_rng(seed)
		self._criterion_options = dict(n_times=n_times, horizon_lengthscales=horizon_lengthscales,
		                               n_max_samples=n_max_samples, n_candidates=n_candidates,
		                               weight=weight, fstar_source=fstar_source,
		                               clip_horizon=clip_horizon)

	def relevance(self, t):
		"""Score every stored observation by its mutual information with the future maximum.

		Args:
				t (float): the present time

		Returns:
				np.array: the relevance of each observation, in nats
		"""
		return mi_criterion(
			self._xx_tt[:, :-1], self._xx_tt[:, -1], self._yy_normalized, t,
			self._lambda, self._lS, max(self._lT, 1e-12), self._noise,
			mean_const=float(self._gpr.mean_module.constant.item()),
			spatial_nu=self._spatial_nu, temporal_nu=self._temporal_nu,
			rng=self._rng, **self._criterion_options)

	def clean(self, t, verbose=False):
		"""Remove observations that carry less information than the budget allows.

		Args:
				t (float): the present time
				verbose (bool, optional): verbose output. Defaults to False.
		"""
		if self._n_initial_observations > 0:
			return

		if self._current_time is None:
			self._current_time = t

		self._budget += self._alpha * (t - self._current_time) / max(self._lT, 1e-12)
		if self._budget_cap is not None:
			self._budget = min(self._budget, self._budget_cap)

		self._last_min_criterion, self._last_min_lT, self._budget_spent = np.nan, np.nan, 0.0
		n_removed = 0

		while self._xx_tt.shape[0] > self._min_dataset_size:
			if self._max_removals_per_clean is not None and n_removed >= self._max_removals_per_clean:
				break

			criteria = self.relevance(t)
			idx_min = int(criteria.argmin())
			min_crit = float(criteria[idx_min])

			# Logged from the first pass only, paired with the lengthscale it was
			# computed under: afterwards it describes a dataset the rest of the run
			# never saw, which is no use for calibrating alpha.
			if n_removed == 0:
				self._last_min_criterion, self._last_min_lT = min_crit, self._lT

			if verbose:
				print(f"Removal Budget: {self._budget} nats // Least Relevant Observation: {idx_min} // "
				      f"Relevancy: {min_crit} nats (i.e. {round(100 * min_crit / self._budget, 2) if self._budget > 0 else float('inf')}% of budget)")

			if min_crit > self._budget:
				break

			# Budget consumption
			self._budget -= min_crit
			self._budget_spent += min_crit
			n_removed += 1

			if verbose:
				print(f"Observation {idx_min} is removed")

			# Dataset update
			self._xx_tt = np.delete(self._xx_tt, (idx_min), axis=0)
			self._yy = np.delete(self._yy, (idx_min), axis=0)
			self.update_surrogate_model(verbose=verbose)

		self._current_time = t
