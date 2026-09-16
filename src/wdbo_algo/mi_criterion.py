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
     candidate set, exactly as in MES (Wang & Jegelka, ICML 2017). One sample
     set per `t`, drawn from the full-`D` posterior and shared across every `i`.

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
from scipy.optimize import brentq
from scipy.special import log_ndtr
from scipy.stats import qmc

SQRT3 = np.sqrt(3.0)
SQRT5 = np.sqrt(5.0)
LOG_2PI = np.log(2.0 * np.pi)

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

	Args:
			mu (np.array): posterior mean of the latent f at the candidate points
			sigma (np.array): posterior standard deviation at the candidate points
			n_samples (int): number of max-value samples to draw
			rng (np.random.Generator): the source of randomness

	Returns:
			np.array: `n_samples` samples of f*
	"""
	sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-8)
	mu = np.asarray(mu, dtype=float)

	def log_cdf(z):
		return log_ndtr((z - mu) / sigma).sum()

	lo = float((mu - 3.0 * sigma).min())
	hi = float((mu + 5.0 * sigma).max())
	# The product of many CDFs is much sharper than any single one, so the
	# textbook bracket can sit entirely on one side of the percentiles. Widen
	# until it straddles them; geometric growth, so this ends quickly.
	span = max(hi - lo, 1e-6)
	for _ in range(64):
		if log_cdf(lo) <= np.log(0.25):
			break
		lo -= span
		span *= 2.0
	span = max(hi - lo, 1e-6)
	for _ in range(64):
		if log_cdf(hi) >= np.log(0.75):
			break
		hi += span
		span *= 2.0

	# Bisect on log P rather than P: the product of 500-odd CDFs underflows to a
	# flat zero well inside the bracket, which leaves brentq nothing to descend.
	q25, q50, q75 = (brentq(lambda z: log_cdf(z) - np.log(p), lo, hi) for p in (0.25, 0.50, 0.75))

	# Percentile matching for Gumbel(a, b): the denominator is negative and
	# q25 - q75 is too, so b comes out positive.
	b = (q25 - q75) / (np.log(np.log(4.0 / 3.0)) - np.log(np.log(4.0)))
	b = max(b, 1e-10)
	a = q50 + b * np.log(np.log(2.0))

	return a - b * np.log(-np.log(rng.random(n_samples)))


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


def mi_criterion(xx, tt, yy, t0, lam, lS, lT, noise_var, mean_const=0.0,
                 spatial_nu=2.5, temporal_nu=1.5, n_times=8, horizon_lengthscales=3.0,
                 n_max_samples=32, n_candidates=512, weight="kernel",
                 clip_horizon=None, rng=None, jitter=1e-8):
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
			clip_horizon (float, optional): cap the future horizon at this absolute
			time. Defaults to None.
			rng (np.random.Generator, optional): the source of randomness.
			jitter (float, optional): initial Cholesky jitter. Defaults to 1e-8.

	Returns:
			np.array: the relevance of each observation, in nats
	"""
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

		# f*_t, sampled once from the full-data posterior and shared across every
		# i. Week 2, 3.1: doing it per leave-one-out posterior instead would cost
		# n.|T| Gumbel fits per clean rather than |T|, for a correction that is
		# rank-one in a posterior conditioned on n points.
		CM = scale * KSM
		mean_m = mean_const + CM.T @ ahat
		var_m = np.maximum(lam - np.einsum("jm,jm->m", CM, A @ CM), 1e-12)
		fstar = sample_max_values_gumbel(mean_m, np.sqrt(var_m), n_max_samples, rng)

		rho2, mean_v, var_v = leave_one_out_moments(A, dA, ahat, scale * KS, lam, mean_const)

		gamma = (fstar[None, :] - mean_v[:, None]) / np.sqrt(var_v)[:, None]
		# -0.5 log(1 - z) via log1p: a stale observation has z ~ 1e-20 or smaller,
		# where forming (1 - z) first would round it away to a relevance of exactly
		# zero. The whole point of the criterion is to rank those observations
		# against each other, so they must not all collapse onto the same value.
		z = rho2[:, None] * truncated_normal_var_deficit(gamma)
		relevance += w_t * (-0.5 * np.log1p(-np.clip(z, 0.0, 1.0 - 1e-12)).mean(axis=1))

	if not np.all(np.isfinite(relevance)):
		raise FloatingPointError("mi_criterion produced non-finite relevance values")

	return np.maximum(relevance, 0.0)
