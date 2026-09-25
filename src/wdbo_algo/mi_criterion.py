"""Mutual-information criterion for removing stale observations in Dynamic BO.

An alternative to `wdbo_criterion.wasserstein_criterion`. W-DBO scores an
observation by how far the whole posterior moves when you drop it; this scores
it by how much it tells you about the quantity the optimizer actually chases --
the instantaneous maximum `f*_t = max_x f(x, t)` at future times `t`:

	R(i) = int_{t0}^{inf} w(t) . I(y_i ; f*_t | D_i) dt,   w(t) prop. to kT(t - t0)

with `D_i = D \\ {(x_i, t_i, y_i)}` and `w` normalized to integrate to 1. The
units are nats, so -- unlike the Wasserstein ratio, which is normalized against
the prior -- R(i) is additive and unnormalized. `MIDBOOptimizer` accrues its
removal budget additively to match.

Estimator (following the MES/OPES line of work):

  1. `f*_t` is sampled by fitting a Gumbel to the max-value CDF over a finite
     candidate set, exactly as in MES (Wang & Jegelka, ICML 2017). By default
     one sample set per `(i, t)`, drawn from the same leave-one-out posterior
     `D_i` that the rest of the term conditions on -- which is what the
     definition of `I(y_i ; f*_t | D_i)` asks for. Passing
     `fstar_source="full"` reverts to one shared sample set per `t` drawn from
     the full-`D` posterior: `|T|` Gumbel fits per clean instead of `n.|T|`,
     at the cost of conditioning `f*_t` on the very observation whose
     information content is being measured. `fstar_source="is"` keeps the
     shared full-`D` sample set but reweights it for each `D_i` by importance
     sampling (Week 3 §3.2, `importance_weights`): `|T|` Gumbel fits, with the
     conditioning approximately undone through the weights.

  2. `H(y_i | f*_t, D_i)` is upper-bounded by the Gaussian entropy of matching
     variance (OPES), which turns the whole problem into computing
     `s_i^2 = Var[y_i | f*_t, D_i]`.

  3. `y_i` (past) and `f*_t` (future) live at different times, so there is no
     direct truncation `y_i <= f*_t` to exploit. The bridge is `v = f(x_i, t)`:
     the latent function at the same place at the future time. `y_i | v` is
     Gaussian, `v | f*_t` is truncated Gaussian, and `y_i` is independent of
     `f*_t` given `v`. The law of total variance then gives a closed form.

See `experiments/MI_CRITERION.md` for the derivation; `mi_criterion` below
states the two results it uses.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import log_ndtr
from scipy.stats import qmc

SQRT3 = np.sqrt(3.0)
SQRT5 = np.sqrt(5.0)
LOG_2PI = np.log(2.0 * np.pi)

# Where the f*_t samples come from; see `mi_criterion`.
FSTAR_SOURCES = ("loo", "full", "is")

# Below this standardized truncation point the direct truncated-variance formula
# is worthless (see _truncated_normal_rel_var).
_ASYMPTOTIC_GAMMA = -30.0


def matern(r, nu):
	"""The Matern correlation function, on an already-scaled distance.

	Matches gpytorch.kernels.MaternKernel so the criterion sees the same kernel
	the surrogate model was fitted with.

	Args:
			r (np.array): distance divided by the lengthscale
			nu (float): smoothness, one of 0.5, 1.5, 2.5

	Returns:
			np.array: the correlation, elementwise
	"""
	if nu == 0.5:
		return np.exp(-r)
	if nu == 1.5:
		a = SQRT3 * r
		return (1.0 + a) * np.exp(-a)
	if nu == 2.5:
		a = SQRT5 * r
		return (1.0 + a + a * a / 3.0) * np.exp(-a)

	raise ValueError(f"Unsupported Matern smoothness nu={nu} (expected 0.5, 1.5 or 2.5)")


def _cdist(a, b):
	"""Euclidean distances between two sets of points.

	Args:
			a (np.array): `n x d` array
			b (np.array): `m x d` array

	Returns:
			np.array: the `n x m` distance matrix
	"""
	diff = a[:, None, :] - b[None, :, :]
	return np.sqrt(np.maximum((diff * diff).sum(axis=-1), 0.0))


def truncated_normal_var_deficit(gamma):
	"""`1 - Var[X | X <= gamma] / Var[X]` for a standard normal X.

	The *deficit* rather than the ratio itself, because that is what `mi_criterion`
	needs and because the two are not equally computable. Writing
	`lam = phi(gamma) / Phi(gamma)`, the textbook variance ratio is
	`1 - gamma.lam - lam^2`, so the deficit is `gamma.lam + lam^2` -- and the
	deficit is the one that survives the regime that matters.

	Three numerical traps, of which only the third is obvious:

	- `phi` and `Phi` both underflow to zero below `gamma ~ -38`, making the ratio
	  0/0. Evaluating it as `exp(log phi - log Phi)` removes that.
	- as `gamma -> -inf` the deficit tends to 1 through two `O(gamma^2)` terms
	  that cancel, so below `_ASYMPTOTIC_GAMMA` it is taken from the expansion
	  `Psi = 1/gamma^2 - 6/gamma^4` instead.
	- as `gamma -> +inf` the deficit tends to *zero*, and this is the regime the
	  removal loop lives in: a stale observation is one whose future value is
	  nowhere near the maximum. Here `gamma.lam + lam^2` is a sum of two positive
	  terms and stays exact all the way down to 1e-300, whereas recovering it as
	  `1 - Psi` would round to zero below 1e-16 -- turning every stale observation
	  into an exact tie and making `argmin` pick among them arbitrarily.

	Args:
			gamma (np.array): the truncation point, standardized

	Returns:
			np.array: the variance deficit, in [0, 1]
	"""
	gamma = np.asarray(gamma, dtype=float)
	out = np.empty(gamma.shape, dtype=float)

	tail = gamma < _ASYMPTOTIC_GAMMA
	g2 = gamma[tail] ** 2
	out[tail] = 1.0 - 1.0 / g2 + 6.0 / (g2 * g2)

	g = gamma[~tail]
	lam = np.exp(-0.5 * g * g - 0.5 * LOG_2PI - log_ndtr(g))
	out[~tail] = g * lam + lam * lam

	return np.clip(out, 0.0, 1.0)


def sample_max_values_gumbel(mu, sigma, n_samples, rng):
	"""Sample the maximum of a GP over a candidate set, by Gumbel approximation.

	The MES recipe: treat the candidate values as independent, so
	`P(f* <= z) = prod_m Phi((z - mu_m) / sigma_m)`; read the 25/50/75 percentiles
	off that CDF by bisection; fit a Gumbel to them; inverse-sample.

	This mirrors `botorch.acquisition.max_value_entropy_search._sample_max_value_Gumbel`,
	which cannot be called directly here: it expects a BoTorch `Posterior` with
	`n x 1`-shaped moments, while `SpaceTimeGPModel` returns a bare gpytorch
	`MultivariateNormal`.

	`mu` and `sigma` may be `r x n_candidates`, in which case the `r` independent
	max-value problems are solved together and `r x n_samples` samples come back.
	That is not a convenience: with `fstar_source="loo"` there is one problem per
	observation per future time, and a Python-level loop over `brentq` dominates
	the whole criterion. The batched path bisects all of them on the same fixed
	iteration count instead, which is one vectorized pass over an `r x n_candidates`
	array per step.

	Args:
			mu (np.array): posterior mean of the latent f at the candidate points,
			either `n_candidates` or `r x n_candidates`
			sigma (np.array): posterior standard deviation, same shape as `mu`
			n_samples (int): number of max-value samples to draw per problem
			rng (np.random.Generator): the source of randomness

	Returns:
			np.array: `n_samples` samples of f*, or `r x n_samples` if `mu` was 2-D
	"""
	mu = np.asarray(mu, dtype=float)
	batched = mu.ndim == 2
	mu = np.atleast_2d(mu)
	sigma = np.maximum(np.atleast_2d(np.asarray(sigma, dtype=float)), 1e-8)
	r = mu.shape[0]

	# The r problems x 3 percentiles are flattened into one list of independent
	# scalar roots, so a solved one can simply be dropped from the work array.
	rows = np.repeat(np.arange(r), 3)
	target = np.tile(np.log(np.array([0.25, 0.50, 0.75])), r)

	def log_cdf(z, act, grad=False):
		"""`log P(f* <= z)` for the active roots, and optionally its derivative."""
		idx = rows[act]
		u = (z[:, None] - mu[idx]) / sigma[idx]
		lcdf = log_ndtr(u)
		if not grad:
			return lcdf.sum(axis=-1), None

		hazard = np.exp(-0.5 * u * u - 0.5 * LOG_2PI - lcdf) / sigma[idx]
		return lcdf.sum(axis=-1), hazard.sum(axis=-1)

	lo = np.repeat((mu - 3.0 * sigma).min(axis=1), 3)
	hi = np.repeat((mu + 5.0 * sigma).max(axis=1), 3)

	# The product of many CDFs is much sharper than any single one, so the
	# textbook bracket can sit entirely on one side of the percentiles. Widen
	# until it straddles them; geometric growth, so this ends quickly.
	everything = np.arange(rows.size)
	span = np.maximum(hi - lo, 1e-6)
	for _ in range(64):
		bad = log_cdf(lo, everything)[0] > target
		if not bad.any():
			break
		lo[bad] -= span[bad]
		span[bad] *= 2.0
	span = np.maximum(hi - lo, 1e-6)
	for _ in range(64):
		bad = log_cdf(hi, everything)[0] < target
		if not bad.any():
			break
		hi[bad] += span[bad]
		span[bad] *= 2.0

	# Solve on log P rather than P: the product of 500-odd CDFs underflows to a
	# flat zero well inside the bracket, which leaves a root finder nothing to
	# descend. log P is monotone in z, so this is safeguarded Newton -- the
	# bracket is kept and a step that would leave it becomes a bisection. Roots
	# drop out of `act` as they converge, which is what keeps the batched solve
	# cheaper than the per-problem one it replaces: with `fstar_source="loo"`
	# there is one problem per observation per future time.
	z = 0.5 * (lo + hi)
	act = everything
	for _ in range(80):
		za, loa, hia = z[act], lo[act], hi[act]
		value, slope = log_cdf(za, act, grad=True)
		residual = value - target[act]

		descend = residual < 0.0
		lo[act] = np.where(descend, za, loa)
		hi[act] = np.where(descend, hia, za)

		done = np.abs(residual) < 1e-10
		if done.all():
			break

		# Step only the roots that are still running: `z` holds the answer, and a
		# root that has converged must keep the value it converged to rather than
		# be moved once more on its way out of `act`.
		act = act[~done]
		za, residual, slope = za[~done], residual[~done], slope[~done]

		newton = za - residual / np.maximum(slope, 1e-300)
		inside = (slope > 0.0) & (newton > lo[act]) & (newton < hi[act])
		z[act] = np.where(inside, newton, 0.5 * (lo[act] + hi[act]))

	q25, q50, q75 = z.reshape(r, 3).T


	# Percentile matching for Gumbel(a, b): the denominator is negative and
	# q25 - q75 is too, so b comes out positive.
	b = (q25 - q75) / (np.log(np.log(4.0 / 3.0)) - np.log(np.log(4.0)))
	b = np.maximum(b, 1e-10)
	a = q50 + b * np.log(np.log(2.0))

	# One set of uniforms, shared by every problem in the batch. The criterion
	# compares its r problems against each other -- with `fstar_source="loo"` they
	# are the n leave-one-out posteriors and the loop takes an argmin over them --
	# so common random numbers cancel most of the Monte-Carlo noise out of that
	# comparison. Each row is still a correct marginal draw from its own Gumbel;
	# only the coupling across rows changes. It also makes the result independent
	# of the order the dataset happens to be stored in, which a per-row draw is not.
	uniform = rng.random(n_samples)[None, :]
	samples = a[:, None] - b[:, None] * np.log(-np.log(uniform))

	return samples if batched else samples[0]


def time_grid(t0, lT, temporal_nu, n_times, horizon_lengthscales, weight, clip_horizon=None):
	"""The quadrature nodes and weights approximating the integral over future times.

	`R(i)` integrates over `[t0, +inf)` against `w(t) prop. to kT(|t - t0|)`. The
	kernel has decayed to a few percent by three lengthscales, so the integral is
	truncated there and discretized with Gauss-Legendre nodes; the weights carry
	the kernel factor.

	`weight="uniform"` drops the kernel factor. Week 2 flags the *double decay* --
	`I(y_i; f*_t)` already vanishes as `t` moves away, and `w(t)` applies the same
	decay a second time -- as a design choice needing empirical support, so both
	are available.

	Args:
			t0 (float): the present time
			lT (float): the temporal lengthscale
			temporal_nu (float): smoothness of the temporal kernel
			n_times (int): number of quadrature nodes
			horizon_lengthscales (float): how many lengthscales of future to cover
			weight (str): "kernel" for w(t) prop. to kT, "uniform" for a flat weight
			clip_horizon (float, optional): cap the horizon at this absolute time.
			Defaults to None (the model's own view of the future, uncapped).

	Returns:
			(np.array, np.array): the times, and weights summing to 1
	"""
	horizon = horizon_lengthscales * lT
	if clip_horizon is not None:
		horizon = min(horizon, max(clip_horizon - t0, 0.0))

	if horizon <= 0.0 or n_times < 1:
		return np.empty(0), np.empty(0)

	nodes, quad_weights = np.polynomial.legendre.leggauss(n_times)
	offsets = 0.5 * horizon * (nodes + 1.0)

	if weight == "kernel":
		quad_weights = quad_weights * matern(offsets / lT, temporal_nu)
	elif weight != "uniform":
		raise ValueError(f"Unknown weight scheme {weight!r} (expected 'kernel' or 'uniform')")

	total = quad_weights.sum()
	if not np.isfinite(total) or total <= 0.0:
		return np.empty(0), np.empty(0)

	return t0 + offsets, quad_weights / total


def _inverse_gram(gram, jitter):
	"""Invert a covariance matrix by Cholesky, adding jitter until it factorizes.

	Args:
			gram (np.array): the `n x n` covariance matrix
			jitter (float): the initial jitter to add to the diagonal

	Returns:
			np.array: the inverse
	"""
	eye = np.eye(gram.shape[0])
	for _ in range(8):
		try:
			return cho_solve(cho_factor(gram + jitter * eye, lower=True), eye)
		except np.linalg.LinAlgError:
			jitter *= 10.0

	raise np.linalg.LinAlgError("Kernel matrix is not positive definite even with jitter")


def leave_one_out_moments(A, dA, ahat, C, lam, mean_const):
	"""The joint moments of `(y_i, v_i)` under `D_i`, for every `i` at once.

	`v_i = f(x_i, t)` is the latent function at observation `i`'s location, at a
	future time `t`. The naive reading of `D_i = D \\ {i}` is `n` separate
	posteriors per future time; it is not needed. With `A = (K + noise.I)^-1` over
	the *full* dataset, `ahat = A (y - mean)` and `C[j, i] = k(z_j, (x_i, t))`,
	set `M = A C` and `w = diag(M)`. Then, exactly:

		Var[y_i | D_i]     = 1 / A_ii                       (the standard GP LOO
		                                                     variance, R&W eq. 5.12)
		Cov(y_i, v_i | D_i) = w_i / A_ii
		Var[v_i | D_i]      = lam - sum_j C_ji M_ji + w_i^2 / A_ii
		E[v_i | D_i]        = mean + (C^T ahat)_i - w_i ahat_i / A_ii

	The `1/A_ii` terms are the rank-one correction that undoes conditioning on
	observation `i`; without them these are the ordinary full-data moments.
	`Var[y_i | D_i]` includes the observation noise, as it must -- `y_i` is what
	would have to be re-measured.

	Only `rho^2` is returned rather than the three variances separately: the
	mutual information depends on the pair `(y_i, v_i)` solely through their
	correlation (see `mi_criterion`), and `Var[y_i | D_i]` cancels out.

	Args:
			A (np.array): `n x n` inverse of the full-data covariance matrix
			dA (np.array): its diagonal
			ahat (np.array): `A @ (y - mean_const)`
			C (np.array): `n x n` cross-covariance, `C[j, i] = k(z_j, (x_i, t))`
			lam (float): the kernel outputscale, i.e. the prior variance
			mean_const (float): the GP's constant mean

	Returns:
			(np.array, np.array, np.array): `rho^2`, `E[v | D_i]` and `Var[v | D_i]`,
			each of length `n`
	"""
	M = A @ C
	w = np.diag(M)

	var_v = np.maximum(lam - np.einsum("ji,ji->i", C, M) + w * w / dA, 1e-12)
	mean_v = mean_const + C.T @ ahat - w * ahat / dA

	# rho^2 = cov^2 / (var_y . var_v) = (w/A_ii)^2 / ((1/A_ii) . var_v).
	# Kept strictly below 1 so the log in mi_criterion stays finite.
	rho2 = np.clip(w * w / (dA * var_v), 0.0, 1.0 - 1e-12)

	return rho2, mean_v, var_v


def full_data_moments(A, ahat, C, lam, mean_const):
	"""The ordinary full-data posterior of `f` at a set of space-time points.

	Used for the candidate set with `fstar_source="full"`, and for the bridge
	variable `v_i = f(x_i, t)` with `fstar_source="is"`, whose weights compare
	`v_i`'s posterior under `D` against the one under `D_i`.

	Args:
			A (np.array): `n x n` inverse of the full-data covariance matrix
			ahat (np.array): `A @ (y - mean_const)`
			C (np.array): `n x m` cross-covariance to the `m` points
			lam (float): the kernel outputscale
			mean_const (float): the GP's constant mean

	Returns:
			(np.array, np.array): the `m` means and variances
	"""
	mean = mean_const + C.T @ ahat
	var = np.maximum(lam - np.einsum("jm,jm->m", C, A @ C), 1e-12)
	return mean, var


def importance_weights(fstar, mean_loo, var_loo, mean_full, var_full):
	"""Self-normalized weights that turn samples of `f*_t | D` into ones of `f*_t | D_i`.

	Week 3 §3.2. By Bayes, with `D = D_i + {y_i}`,
	`p(f* | D_i) / p(f* | D) = p(y_i | D_i) / p(y_i | f*, D_i)`. Replacing the
	conditioning on `f*` by the event `v_i <= f*` (assumption c) and treating
	`y_i` as independent of `f*` given `v_i` (assumption d) turns the ratio into

		w_i(f*) = Phi(gamma_loo) / Phi(gamma_full) = P(v_i <= f* | D_i) / P(v_i <= f* | D)

	with `gamma = (f* - E[v_i]) / sd[v_i]` under each posterior. Because both
	assumptions are approximations, the weights do not average to one under
	`f* | D`, so each row is normalized to sum to one over the sample set.

	The ratio is formed in log space: a sample `f*` sitting far below `E[v_i | D]`
	makes both `Phi`s underflow, and their ratio is then 0/0 in float64.

	Args:
			fstar (np.array): the `K` shared samples of `f*_t`, drawn under `D`
			mean_loo (np.array): `E[v_i | D_i]`, length `n`
			var_loo (np.array): `Var[v_i | D_i]`, length `n`
			mean_full (np.array): `E[v_i | D]`, length `n`
			var_full (np.array): `Var[v_i | D]`, length `n`

	Returns:
			np.array: `n x K` weights, each row non-negative and summing to 1
	"""
	fstar = np.asarray(fstar, dtype=float).ravel()[None, :]
	gamma_loo = (fstar - mean_loo[:, None]) / np.sqrt(var_loo)[:, None]
	gamma_full = (fstar - mean_full[:, None]) / np.sqrt(var_full)[:, None]

	log_w = log_ndtr(gamma_loo) - log_ndtr(gamma_full)
	w = np.exp(log_w - log_w.max(axis=1, keepdims=True))
	return w / w.sum(axis=1, keepdims=True)


def leave_one_out_candidate_moments(A, dA, ahat, CM, lam, mean_const):
	"""The posterior of `f(x_m, t)` at every candidate `m` under `D_i`, for every `i`.

	The same rank-one correction as `leave_one_out_moments`, applied to the
	candidate set instead of to the observations' own locations. With `M = A CM`
	and `mu`, `s2` the full-data moments at the candidates:

		E[f_m | D_i]   = mu_m - M_im . ahat_i / A_ii
		Var[f_m | D_i] = s2_m + M_im^2 / A_ii

	The variance *grows*, as it must: dropping an observation can only make the
	posterior less certain. Forming the `n` candidate posteriors this way costs
	one `n x n_candidates` matrix product, not `n` refits.

	Args:
			A (np.array): `n x n` inverse of the full-data covariance matrix
			dA (np.array): its diagonal
			ahat (np.array): `A @ (y - mean_const)`
			CM (np.array): `n x n_candidates` cross-covariance to the candidates
			lam (float): the kernel outputscale
			mean_const (float): the GP's constant mean

	Returns:
			(np.array, np.array): `n x n_candidates` means and variances, row `i`
			holding the posterior under `D_i`
	"""
	M = A @ CM
	mu = mean_const + CM.T @ ahat
	s2 = lam - np.einsum("jm,jm->m", CM, M)

	mean_loo = mu[None, :] - M * (ahat / dA)[:, None]
	var_loo = np.maximum(s2[None, :] + M * M / dA[:, None], 1e-12)

	return mean_loo, var_loo


def mi_criterion(xx, tt, yy, t0, lam, lS, lT, noise_var, mean_const=0.0,
                 spatial_nu=2.5, temporal_nu=1.5, n_times=8, horizon_lengthscales=3.0,
                 n_max_samples=32, n_candidates=512, weight="kernel",
                 fstar_source="loo", clip_horizon=None, candidates=None, rng=None, jitter=1e-8):
	"""Score every observation by its mutual information with the future maximum.

	The returned relevance is in nats and is non-negative by construction. It is a
	*lower* bound on the true mutual information, since step 2 of the estimator
	upper-bounds the entropy being subtracted -- the safe direction for a removal
	rule, which can then understate but never overstate a point's importance.

	Per future time `t`, combining the truncated-Gaussian `v | f*_t` with the
	Gaussian `y_i | v` through the law of total variance gives

		Var[y_i | f*_t, D_i] = Var[y_i | D_i] . [1 - rho_i^2 . (1 - Psi(gamma))]

	with `gamma = (f*_t - E[v_i]) / sd[v_i]` and `Psi` the relative variance of the
	truncated normal. The prefactor is exactly `Var[y_i | D_i]`, which is also what
	the unconditioned entropy contributes, so it cancels and leaves

		I(y_i ; f*_t | D_i) ~ -0.5 . mean over f*_t of log[1 - rho_i^2 (1 - Psi)]

	i.e. OPES's `log sigma - mean log s`, with only the correlation between the
	observation and the future function value at its own location surviving.

	Args:
			xx (np.array): `n x d` spatial inputs, normalized to [0, 1]^d
			tt (np.array): the `n` observation times
			yy (np.array): the `n` observations, standardized like the GP saw them
			t0 (float): the present time
			lam (float): the kernel outputscale
			lS (float): the spatial lengthscale
			lT (float): the temporal lengthscale
			noise_var (float): the observation noise variance
			mean_const (float, optional): the GP's constant mean. Defaults to 0.0.
			spatial_nu (float, optional): spatial Matern smoothness. Defaults to 2.5.
			temporal_nu (float, optional): temporal Matern smoothness. Defaults to 1.5.
			n_times (int, optional): quadrature nodes over future time. Defaults to 8.
			horizon_lengthscales (float, optional): how far ahead to integrate, in
			temporal lengthscales. Defaults to 3.0.
			n_max_samples (int, optional): Monte-Carlo samples of f*_t. Defaults to 32.
			n_candidates (int, optional): candidate points discretizing X for the
			max-value CDF. Defaults to 512.
			weight (str, optional): "kernel" or "uniform". Defaults to "kernel".
			fstar_source (str, optional): which posterior `f*_t` is sampled from.
			"loo" draws a separate sample set from each leave-one-out posterior
			`D_i`, as the definition of the conditional mutual information requires;
			"full" draws one set from the full-`D` posterior and shares it across
			every `i`, which is `n` times cheaper but conditions `f*_t` on the
			observation being scored. "is" draws the same shared set as "full" and
			reweights it per `i` towards `D_i` by importance sampling (Week 3 §3.2,
			see `importance_weights`), at roughly the cost of "full". Defaults to "loo".
			clip_horizon (float, optional): cap the future horizon at this absolute
			time. Defaults to None.
			candidates (np.array, optional): `n_candidates x d` points in [0, 1]^d
			discretizing X for the max-value CDF. Defaults to None, a scrambled
			Sobol set of `n_candidates` points drawn from `rng`. Fixing it isolates
			the Monte-Carlo error of the `f*_t` samples from the discretization.
			rng (np.random.Generator, optional): the source of randomness.
			jitter (float, optional): initial Cholesky jitter. Defaults to 1e-8.

	Returns:
			np.array: the relevance of each observation, in nats
	"""
	if fstar_source not in FSTAR_SOURCES:
		raise ValueError(f"Unknown fstar_source {fstar_source!r} (expected one of {FSTAR_SOURCES})")

	rng = np.random.default_rng() if rng is None else rng
	xx = np.asarray(xx, dtype=float)
	tt = np.asarray(tt, dtype=float).ravel()
	yy = np.asarray(yy, dtype=float).ravel()
	n, d = xx.shape

	times, omega = time_grid(t0, lT, temporal_nu, n_times, horizon_lengthscales, weight, clip_horizon)
	if times.size == 0:
		return np.zeros(n)

	# The kernel is separable, k((x,t),(x',t')) = lam . kS(x,x') . kT(t,t'), and
	# every point the criterion evaluates sits at a spatial location that does not
	# depend on t -- either an observation's own x, or a candidate. So both spatial
	# Grams are built once here, and each future time is a row rescaling of them.
	KS = matern(_cdist(xx, xx) / lS, spatial_nu)
	if candidates is None:
		candidates = qmc.Sobol(d, scramble=True, seed=rng).random(n_candidates)
	KSM = matern(_cdist(xx, candidates) / lS, spatial_nu)

	KT = matern(np.abs(tt[:, None] - tt[None, :]) / lT, temporal_nu)
	A = _inverse_gram(lam * KS * KT + noise_var * np.eye(n), jitter)
	dA = np.maximum(np.diag(A), 1e-12)
	ahat = A @ (yy - mean_const)

	relevance = np.zeros(n)
	for t, w_t in zip(times, omega):
		# The t-dependent factor of the cross-covariances, shared by both Grams.
		scale = (lam * matern(np.abs(tt - t) / lT, temporal_nu))[:, None]

		CM = scale * KSM
		C = scale * KS
		if fstar_source == "loo":
			# f*_t drawn from each D_i in turn, so the sample set entering
			# I(y_i ; f*_t | D_i) is conditioned on the same data the rest of the
			# term is. The moments are a rank-one correction of the full-data ones
			# (one matrix product for all i); only the n Gumbel fits are extra.
			mean_loo, var_loo = leave_one_out_candidate_moments(A, dA, ahat, CM, lam, mean_const)
			fstar = sample_max_values_gumbel(mean_loo, np.sqrt(var_loo), n_max_samples, rng)
		else:
			# One sample set from the full-data posterior, shared across every i:
			# |T| Gumbel fits per clean rather than n.|T|. Used as-is ("full"), it
			# lets observation i inform the f*_t it is being scored against; "is"
			# reweights it towards D_i below instead.
			mean_m, var_m = full_data_moments(A, ahat, CM, lam, mean_const)
			fstar = sample_max_values_gumbel(mean_m, np.sqrt(var_m), n_max_samples, rng)[None, :]

		rho2, mean_v, var_v = leave_one_out_moments(A, dA, ahat, C, lam, mean_const)

		gamma = (fstar - mean_v[:, None]) / np.sqrt(var_v)[:, None]
		# -0.5 log(1 - z) via log1p: a stale observation has z ~ 1e-20 or smaller,
		# where forming (1 - z) first would round it away to a relevance of exactly
		# zero. The whole point of the criterion is to rank those observations
		# against each other, so they must not all collapse onto the same value.
		z = rho2[:, None] * truncated_normal_var_deficit(gamma)
		mi = -0.5 * np.log1p(-np.clip(z, 0.0, 1.0 - 1e-12))

		if fstar_source == "is":
			mean_full, var_full = full_data_moments(A, ahat, C, lam, mean_const)
			weights = importance_weights(fstar, mean_v, var_v, mean_full, var_full)
			relevance += w_t * (weights * mi).sum(axis=1)
		else:
			relevance += w_t * mi.mean(axis=1)

	if not np.all(np.isfinite(relevance)):
		raise FloatingPointError("mi_criterion produced non-finite relevance values")

	return np.maximum(relevance, 0.0)
