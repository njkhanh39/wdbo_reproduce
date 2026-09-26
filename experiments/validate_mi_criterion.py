"""Check `wdbo_algo.mi_criterion` against brute-force references.

The criterion buys its speed with two shortcuts that are easy to get subtly
wrong and impossible to spot from a finished experiment:

- the `n` leave-one-out posteriors are never formed; they are read off the
  full-data inverse by a rank-one correction (`leave_one_out_moments`);
- `Var[y_i | f*, D_i]` is a closed form standing in for an integral over a
  truncated Gaussian, evaluated in a regime where the textbook expression
  cancels to nothing (`_truncated_normal_rel_var`).

Both have exact or Monte-Carlo references, so both are checked here rather than
inferred from whether a 10-minute run "looked reasonable". A wrong criterion
does not crash -- it deletes the wrong observations and reports a plausible
regret curve.

Usage:
    python experiments/validate_mi_criterion.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import truncnorm

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from wdbo_algo.mi_criterion import (  # noqa: E402
    full_data_moments,
    importance_weights,
    leave_one_out_candidate_moments,
    leave_one_out_moments,
    matern,
    mi_criterion,
    sample_max_values_gumbel,
    time_grid,
    truncated_normal_var_deficit,
)

# The hyperparameters a fitted SpaceTimeGPModel would hand the criterion. Chosen
# so the temporal lengthscale is comparable to the spread of observation times --
# the regime the removal loop actually runs in.
LAM, LS, LT, NOISE = 1.7, 0.35, 0.25, 0.04
SPATIAL_NU, TEMPORAL_NU = 2.5, 1.5

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = ""):
    (PASSED if ok else FAILED).append(name)
    print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def kernel(xa, ta, xb, tb):
    """The separable space-time kernel, built the long way for reference use."""
    dx = np.sqrt(np.maximum(((xa[:, None, :] - xb[None, :, :]) ** 2).sum(-1), 0.0))
    dt = np.abs(ta[:, None] - tb[None, :])
    return LAM * matern(dx / LS, SPATIAL_NU) * matern(dt / LT, TEMPORAL_NU)


def dataset(n, d, seed):
    rng = np.random.default_rng(seed)
    return rng.random((n, d)), np.sort(rng.random(n)), rng.normal(size=n)


# --------------------------------------------------------------------------
# 1. The leave-one-out identities
# --------------------------------------------------------------------------

def test_leave_one_out_moments():
    """Compare against explicitly refitting the GP on D minus {i}.

    This is an exact identity, not an approximation, so the tolerance is
    numerical rather than statistical.
    """
    print("\n1. Leave-one-out moments vs. explicit refit on D \\ {i}")
    n, d, mean_const, t = 11, 3, 0.4, 0.83
    xx, tt, yy = dataset(n, d, seed=1)

    A = np.linalg.inv(kernel(xx, tt, xx, tt) + NOISE * np.eye(n))
    dA = np.diag(A)
    ahat = A @ (yy - mean_const)
    C = kernel(xx, tt, xx, np.full(n, t))
    rho2, mean_v, var_v = leave_one_out_moments(A, dA, ahat, C, LAM, mean_const)

    errors = {"var_y": 0.0, "var_v": 0.0, "cov": 0.0, "mean_v": 0.0}
    for i in range(n):
        keep = [j for j in range(n) if j != i]
        xi, ti, yi = xx[keep], tt[keep], yy[keep] - mean_const
        Ai = np.linalg.inv(kernel(xi, ti, xi, ti) + NOISE * np.eye(n - 1))

        zx, zt = xx[i:i + 1], tt[i:i + 1]
        vx, vt = xx[i:i + 1], np.array([t])
        kz, kv = kernel(xi, ti, zx, zt), kernel(xi, ti, vx, vt)

        ref_var_y = (LAM - kz.T @ Ai @ kz).item() + NOISE
        ref_var_v = (LAM - kv.T @ Ai @ kv).item()
        ref_cov = (kernel(zx, zt, vx, vt) - kz.T @ Ai @ kv).item()
        ref_mean_v = mean_const + (kv.T @ Ai @ yi).item()

        # The criterion only ever needs rho^2 and the moments of v, so var_y and
        # cov are recovered from it the way the derivation defines them.
        errors["var_y"] = max(errors["var_y"], abs(ref_var_y - 1.0 / dA[i]))
        errors["var_v"] = max(errors["var_v"], abs(ref_var_v - var_v[i]))
        errors["mean_v"] = max(errors["mean_v"], abs(ref_mean_v - mean_v[i]))
        errors["cov"] = max(errors["cov"], abs(ref_cov ** 2 / (ref_var_y * ref_var_v) - rho2[i]))

    for name, err in errors.items():
        check(f"{name} matches refit", err < 1e-9, f"max abs error {err:.2e}")


def test_leave_one_out_candidate_moments():
    """Same check for the candidate posteriors that `fstar_source="loo"` samples from.

    With `f*_t` drawn per leave-one-out posterior, the whole default path rests
    on this second rank-one identity; get it wrong and `f*_t` is sampled from a
    posterior that is merely close to `D_i`, which no downstream test would catch.
    """
    print("\n1b. Leave-one-out candidate moments vs. explicit refit")
    n, d, m, mean_const, t = 11, 3, 9, 0.4, 0.83
    xx, tt, yy = dataset(n, d, seed=1)
    candidates = np.random.default_rng(5).random((m, d))

    A = np.linalg.inv(kernel(xx, tt, xx, tt) + NOISE * np.eye(n))
    dA = np.diag(A)
    ahat = A @ (yy - mean_const)
    CM = kernel(xx, tt, candidates, np.full(m, t))
    mean_loo, var_loo = leave_one_out_candidate_moments(A, dA, ahat, CM, LAM, mean_const)

    err_mean = err_var = 0.0
    for i in range(n):
        keep = [j for j in range(n) if j != i]
        Ai = np.linalg.inv(kernel(xx[keep], tt[keep], xx[keep], tt[keep]) + NOISE * np.eye(n - 1))
        kv = kernel(xx[keep], tt[keep], candidates, np.full(m, t))

        ref_mean = mean_const + kv.T @ Ai @ (yy[keep] - mean_const)
        ref_var = LAM - np.einsum("jm,jm->m", kv, Ai @ kv)
        err_mean = max(err_mean, float(np.abs(ref_mean - mean_loo[i]).max()))
        err_var = max(err_var, float(np.abs(ref_var - var_loo[i]).max()))

    check("candidate mean matches refit", err_mean < 1e-9, f"max abs error {err_mean:.2e}")
    check("candidate variance matches refit", err_var < 1e-9, f"max abs error {err_var:.2e}")
    # Dropping data cannot sharpen the posterior; the correction is +M^2/A_ii.
    full_var = LAM - np.einsum("jm,jm->m", CM, A @ CM)
    check("dropping an observation never lowers the candidate variance",
          bool(np.all(var_loo >= full_var[None, :] - 1e-12)))


def test_permutation_invariance():
    """Reordering the dataset must not reorder which point looks removable."""
    print("\n2. Invariance to dataset ordering")
    n, d = 14, 3
    xx, tt, yy = dataset(n, d, seed=2)
    perm = np.random.default_rng(7).permutation(n)

    kwargs = dict(t0=1.0, lam=LAM, lS=LS, lT=LT, noise_var=NOISE, mean_const=0.2,
                  n_times=4, n_max_samples=64, n_candidates=128)
    # Same seed on both sides: the estimator is Monte-Carlo, so only its
    # deterministic part is order-invariant.
    base = mi_criterion(xx, tt, yy, rng=np.random.default_rng(0), **kwargs)
    shuffled = mi_criterion(xx[perm], tt[perm], yy[perm], rng=np.random.default_rng(0), **kwargs)

    err = np.abs(base[perm] - shuffled).max()
    check("relevance is permutation-equivariant", err < 1e-9, f"max abs error {err:.2e}")


# --------------------------------------------------------------------------
# 3. The truncated-variance factor
# --------------------------------------------------------------------------

def test_truncated_variance():
    """Check the deficit against scipy in the mild range and its asymptotes outside it."""
    print("\n3. Truncated-normal variance deficit")
    mild = np.linspace(-6.0, 6.0, 61)
    ref = 1.0 - np.array([truncnorm.var(-np.inf, g) for g in mild])
    err = np.abs(ref - truncated_normal_var_deficit(mild)).max()
    check("matches scipy.stats.truncnorm on [-6, 6]", err < 1e-9, f"max abs error {err:.2e}")

    # Below about -38 the naive phi/Phi ratio is 0/0 in float64. The tail branch
    # must reproduce the asymptote Psi -> 1/gamma^2, i.e. deficit -> 1 - 1/gamma^2.
    deep = np.array([-20.0, -50.0, -200.0, -1e3, -1e6])
    deficit = truncated_normal_var_deficit(deep)
    rel = np.abs((1.0 - deficit) - 1.0 / deep ** 2) * deep ** 2
    check("finite and in [0, 1] deep in the left tail",
          bool(np.all(np.isfinite(deficit)) and np.all(deficit >= 0)), f"deficit(-1e6) = {deficit[-1]:.6f}")
    check("approaches 1 - 1/gamma^2 in the left tail", bool(np.all(rel < 0.02)), f"max rel error {rel.max():.2e}")

    # The regime the removal loop lives in. A stale observation has f* sitting
    # many sigma above its own future value, so the deficit must stay accurate as
    # it decays -- not round to zero, which would tie every stale point together.
    far = np.array([8.0, 12.0, 20.0, 30.0])
    deficit = truncated_normal_var_deficit(far)
    check("stays strictly positive and ordered in the right tail",
          bool(np.all(deficit > 0.0) and np.all(np.diff(deficit) < 0.0)),
          "  ".join(f"g={g:.0f}: {v:.2e}" for g, v in zip(far, deficit)))
    # phi(30)*30 ~ 1.4e-195: far below the 1e-16 at which `1 - Psi` would vanish.
    check("resolves deficits below 1e-16", float(deficit[-1]) < 1e-100 and float(deficit[-1]) > 0.0)

    grid = np.linspace(-40.0, 8.0, 2000)
    check("monotone decreasing in gamma", bool(np.all(np.diff(truncated_normal_var_deficit(grid)) <= 1e-12)))


def test_conditional_variance_monte_carlo():
    """Check s^2 = Var[y_i | f*, D_i] against sampling the bridge variable v.

    The closed form is the law of total variance applied to `y_i` over `v`, so
    the reference draws `v` from its truncated posterior, draws `y_i | v`, and
    takes the sample variance. Independent of every identity in section 1.
    """
    print("\n4. Var[y_i | f*, D_i] vs. Monte Carlo over the bridge variable")
    rng = np.random.default_rng(11)
    n_draws = 2_000_000

    for var_y, var_v, rho, gamma in [(0.9, 1.4, 0.8, 1.2), (0.3, 0.7, 0.35, -0.4),
                                     (1.1, 2.0, 0.95, -2.5), (0.6, 0.6, 0.15, 3.0)]:
        cov = rho * np.sqrt(var_y * var_v)
        closed = var_y * (1.0 - rho ** 2 * truncated_normal_var_deficit(np.array([gamma]))[0])

        # v | f* is N(m_v, var_v) truncated above at f*, i.e. standardized to
        # (-inf, gamma]; y_i | v is Gaussian with a v-independent variance.
        v = truncnorm.rvs(-np.inf, gamma, size=n_draws, random_state=rng) * np.sqrt(var_v)
        y = (cov / var_v) * v + rng.normal(0.0, np.sqrt(var_y - cov ** 2 / var_v), n_draws)

        rel = abs(y.var() - closed) / closed
        check(f"rho={rho}, gamma={gamma:+.1f}", rel < 5e-3,
              f"closed {closed:.6f} vs MC {y.var():.6f} ({rel:.2%})")


# --------------------------------------------------------------------------
# 5. Sampling and quadrature
# --------------------------------------------------------------------------

def test_gumbel_sampler():
    """The samples must reproduce the quartiles of the CDF the Gumbel was fitted to.

    The Gumbel is fitted by matching the 25/50/75 percentiles of
    `prod_m Phi((z - mu_m) / sigma_m)`, so those three points are the one thing
    the sampler is required to get right. Computing them here independently
    checks the percentile matching *and* the inverse sampling in one go.
    """
    print("\n5. Gumbel max-value sampling")
    rng = np.random.default_rng(3)
    mu, sigma = rng.normal(0.0, 1.0, 400), rng.uniform(0.2, 0.8, 400)
    samples = sample_max_values_gumbel(mu, sigma, 400_000, rng)

    check("samples are finite", bool(np.all(np.isfinite(samples))))
    # Under the independence approximation the max of 400 candidates sits above
    # the best of their means; below it would mean a sign or scale error.
    check("f* sits above the candidate means", float(np.median(samples)) > float(mu.max()),
          f"median f* {np.median(samples):.3f} vs max mu {mu.max():.3f}")

    from scipy.optimize import brentq
    from scipy.special import log_ndtr

    def product_cdf_quantile(p):
        return brentq(lambda z: log_ndtr((z - mu) / sigma).sum() - np.log(p), -20.0, 20.0)

    for p in (0.25, 0.50, 0.75):
        target = product_cdf_quantile(p)
        empirical = float(np.quantile(samples, p))
        check(f"sample quantile p={p:.2f} matches the max-value CDF", abs(empirical - target) < 0.02,
              f"target {target:.4f} vs empirical {empirical:.4f}")

    # The batched path is the one `fstar_source="loo"` uses, and it solves its
    # percentiles by a different route (safeguarded Newton over an active set,
    # not one brentq per problem), so it is checked against the 1-D path rather
    # than assumed to follow from it.
    mus = np.stack([mu, mu + 1.5, mu * 0.5 - 2.0])
    sigmas = np.stack([sigma, sigma * 2.0, sigma * 0.3])
    batched = sample_max_values_gumbel(mus, sigmas, 200_000, np.random.default_rng(4))
    check("batched sampling returns one row per problem", batched.shape == (3, 200_000))

    err = 0.0
    for row in range(3):
        single = sample_max_values_gumbel(mus[row], sigmas[row], 200_000, np.random.default_rng(4))
        for p in (0.25, 0.50, 0.75):
            err = max(err, abs(float(np.quantile(batched[row], p)) - float(np.quantile(single, p))))
    check("batched rows match the single-problem sampler", err < 0.01, f"max quantile gap {err:.2e}")


def test_time_grid():
    """Weights sum to 1, cover the intended horizon, and decay like the kernel."""
    print("\n6. Time grid and weights")
    times, omega = time_grid(0.5, LT, TEMPORAL_NU, 8, 3.0, "kernel")
    check("weights sum to 1", abs(omega.sum() - 1.0) < 1e-12)
    check("nodes lie in (t0, t0 + 3 lT)", bool(np.all(times > 0.5) and np.all(times < 0.5 + 3 * LT)))
    check("kernel weighting front-loads the near future", omega[0] > omega[-1],
          f"first {omega[0]:.3f} vs last {omega[-1]:.3f}")

    _, flat = time_grid(0.5, LT, TEMPORAL_NU, 8, 3.0, "uniform")
    check("uniform weighting does not", abs(flat[0] - flat[-1]) < 1e-12)

    clipped, _ = time_grid(0.95, LT, TEMPORAL_NU, 8, 3.0, "kernel", clip_horizon=1.0)
    check("clip_horizon caps the grid", bool(np.all(clipped <= 1.0)), f"max t {clipped.max():.4f}")
    empty, _ = time_grid(1.0, LT, TEMPORAL_NU, 8, 3.0, "kernel", clip_horizon=1.0)
    check("an empty horizon yields no nodes", empty.size == 0)


# --------------------------------------------------------------------------
# 7. End-to-end behaviour
# --------------------------------------------------------------------------

def test_relevance_behaviour():
    """Sanity properties the criterion must have to be usable as a removal rule."""
    print("\n7. Relevance behaviour")
    n, d = 30, 3
    rng = np.random.default_rng(5)
    xx, tt = rng.random((n, d)), np.sort(rng.uniform(0.0, 1.0, n))
    yy = rng.normal(size=n)

    kwargs = dict(lam=LAM, lS=LS, lT=LT, noise_var=NOISE, mean_const=0.0,
                  n_times=8, n_max_samples=64, n_candidates=256)
    r = mi_criterion(xx, tt, yy, t0=1.0, rng=np.random.default_rng(0), **kwargs)

    check("finite", bool(np.all(np.isfinite(r))))
    check("non-negative", bool(np.all(r >= 0.0)), f"min {r.min():.3e}")
    check("not identically zero", float(r.max()) > 0.0, f"max {r.max():.3e} nats")

    # The whole premise of removal: an observation loses relevance as the present
    # moves away from it. Compare the oldest third against the newest third.
    old, new = r[:n // 3].mean(), r[-n // 3:].mean()
    check("stale observations score below recent ones", old < new,
          f"oldest third {old:.4e} vs newest third {new:.4e} nats")

    # Late in a real run lT collapses to ~0.09 and the whole dataset is several
    # lengthscales stale, which is where the criterion underflowed to a column of
    # exact zeros before log1p. Ties there are not a cosmetic problem: argmin
    # would pick among them arbitrarily, so the removal loop would stop ranking.
    stale = mi_criterion(xx, tt, yy, t0=1.0, rng=np.random.default_rng(0),
                         **(kwargs | {"lT": 0.05}))
    check("stale datasets keep strictly positive relevance", bool(np.all(stale > 0.0)),
          f"min {stale.min():.3e} nats at lT=0.05")
    check("stale datasets produce no ties", len(np.unique(stale)) == n,
          f"{len(np.unique(stale))}/{n} distinct values")

    # Same check along the other axis: an observation duplicated in space and
    # time is redundant, so at least one copy should score near zero.
    xd = np.vstack([xx, xx[-1] + 1e-6])
    td, yd = np.append(tt, tt[-1] + 1e-6), np.append(yy, yy[-1])
    rd = mi_criterion(xd, td, yd, t0=1.0, rng=np.random.default_rng(0), **kwargs)
    check("a duplicated observation scores near zero", float(min(rd[-1], rd[-2])) < 0.05 * float(r.max()),
          f"duplicate pair {rd[-2]:.3e}, {rd[-1]:.3e} nats vs max {r.max():.3e}")


def test_importance_weights():
    """The pieces `fstar_source="is"` adds on top of the shared full-data samples.

    The weights `Phi(gamma_loo) / Phi(gamma_full)` rest on one exact step (Week 3
    eq. 15): conditioning the leave-one-out pair `(y_i, v_i)` on the observed
    `y_i` must give back `v_i`'s full-data posterior. That is checked as an
    identity between `leave_one_out_moments` and `full_data_moments`, and
    `full_data_moments` itself against the textbook GP formula. What is not
    checked here -- because it is an approximation, not an identity -- is how
    close the reweighted samples get to `f*_t | D_i`; that is what
    `compare_fstar_sampling.py` measures.
    """
    print("\n8. Importance-sampling weights (fstar_source='is')")
    n, d, mean_const, t = 11, 3, 0.4, 0.83
    xx, tt, yy = dataset(n, d, seed=1)

    A = np.linalg.inv(kernel(xx, tt, xx, tt) + NOISE * np.eye(n))
    dA = np.diag(A)
    ahat = A @ (yy - mean_const)
    C = kernel(xx, tt, xx, np.full(n, t))
    _, mean_loo, var_loo = leave_one_out_moments(A, dA, ahat, C, LAM, mean_const)
    mean_full, var_full = full_data_moments(A, ahat, C, LAM, mean_const)

    ref_mean = mean_const + C.T @ A @ (yy - mean_const)
    ref_var = LAM - np.einsum("ji,ji->i", C, A @ C)
    err = max(np.abs(ref_mean - mean_full).max(), np.abs(ref_var - var_full).max())
    check("full-data moments of v match the GP formula", err < 1e-9, f"max abs error {err:.2e}")

    # Bayes on the bivariate Gaussian (y_i, v_i) | D_i: Var[y_i] = 1/A_ii,
    # Cov = w_i/A_ii, and the LOO residual y_i - E[y_i | D_i] = ahat_i/A_ii, so
    # the regression coefficient Cov/Var[y_i] is w_i.
    w = np.diag(A @ C)
    bayes_mean = mean_loo + w * (ahat / dA)
    bayes_var = var_loo - w * w / dA
    err = max(np.abs(bayes_mean - mean_full).max(), np.abs(bayes_var - var_full).max())
    check("conditioning (y_i, v_i) | D_i on y_i recovers v_i | D (Week 3 eq. 15)", err < 1e-9,
          f"max abs error {err:.2e}")

    rng = np.random.default_rng(6)
    fstar = rng.normal(2.0, 0.5, 64)
    weights = importance_weights(fstar, mean_loo, var_loo, mean_full, var_full)
    check("weights are non-negative and each row sums to 1",
          bool(np.all(weights >= 0.0) and np.allclose(weights.sum(axis=1), 1.0)))

    from scipy.stats import norm
    raw = norm.cdf((fstar[None, :] - mean_loo[:, None]) / np.sqrt(var_loo)[:, None]) \
        / norm.cdf((fstar[None, :] - mean_full[:, None]) / np.sqrt(var_full)[:, None])
    err = np.abs(raw / raw.sum(axis=1, keepdims=True) - weights).max()
    check("weights match the direct Phi ratio where it is computable", err < 1e-12, f"max abs error {err:.2e}")

    same = importance_weights(fstar, mean_full, var_full, mean_full, var_full)
    check("identical posteriors give uniform weights", bool(np.allclose(same, 1.0 / fstar.size)))

    # A sample 60 sd below both means: each Phi is ~1e-790, so the ratio is 0/0
    # unless it is formed in log space.
    deep = importance_weights(np.array([-60.0, 0.0, 3.0]), np.zeros(1), np.ones(1), np.full(1, 0.3), np.full(1, 0.8))
    check("finite when both Phi underflow", bool(np.all(np.isfinite(deep))), f"weights {np.round(deep[0], 4)}")

    # End-to-end: the weights only reweight, so `is` must stay a valid relevance,
    # and for observations whose own future value sits far below f*_t they are
    # ~1 and `is` should coincide with `full` drawn from the same randoms.
    n, d = 30, 3
    xx, tt, yy = dataset(n, d, seed=5)
    kwargs = dict(t0=1.0, lam=LAM, lS=LS, lT=LT, noise_var=NOISE, n_times=6, n_max_samples=64, n_candidates=256)
    full = mi_criterion(xx, tt, yy, fstar_source="full", rng=np.random.default_rng(0), **kwargs)
    imp = mi_criterion(xx, tt, yy, fstar_source="is", rng=np.random.default_rng(0), **kwargs)
    check("is: finite and non-negative", bool(np.all(np.isfinite(imp)) and np.all(imp >= 0.0)))
    rel = np.abs(imp - full) / np.maximum(full, 1e-300)
    check("is coincides with full for the stale half", float(rel[:n // 2].max()) < 1e-3,
          f"max rel gap {rel[:n // 2].max():.2e} (all: {rel.max():.2e})")

    perm = np.random.default_rng(7).permutation(n)
    shuffled = mi_criterion(xx[perm], tt[perm], yy[perm], fstar_source="is", rng=np.random.default_rng(0), **kwargs)
    err = np.abs(imp[perm] - shuffled).max()
    check("is is permutation-equivariant", err < 1e-9, f"max abs error {err:.2e}")

    try:
        mi_criterion(xx, tt, yy, fstar_source="bogus", **kwargs)
        check("unknown fstar_source is rejected", False)
    except ValueError:
        check("unknown fstar_source is rejected", True)


def test_cost():
    """The criterion is called inside the cleaning loop, so it has a time budget.

    W-DBO's own cleaning averages ~0.49 s per query on ackley4d (see
    data/synthetic/ackley4d/results_t-32_32), and that covers a whole removal
    loop -- several criterion evaluations. A single evaluation therefore has to
    land well under that, or the two arms are no longer comparable at a fixed
    wall-clock budget.
    """
    print("\n9. Cost at realistic dataset sizes")
    for n in (50, 150, 250):
        xx, tt, yy = dataset(n, 3, seed=n)
        for source, budget in (("full", 0.15), ("is", 0.25), ("loo", 12.0)):
            mark = time.time()
            mi_criterion(xx, tt, yy, t0=1.0, lam=LAM, lS=LS, lT=LT, noise_var=NOISE,
                         n_times=8, n_max_samples=32, n_candidates=512,
                         fstar_source=source, rng=np.random.default_rng(0))
            elapsed = time.time() - mark
            check(f"n={n} fstar_source={source} under {budget:g} s", elapsed < budget,
                  f"{elapsed * 1000:.1f} ms")

    # The default costs one max-value solve per observation rather than one in
    # total, so it is the best part of n times dearer and the budget above is
    # loose. At a fixed wall-clock experiment budget that buys fewer queries,
    # which is the trade --mi-fstar full/is exist to undo; the timings are printed
    # so that choice rests on measurements and not on the ratio being "about n".


def main():
    # Underflow is routine and harmless here -- exp(-800) legitimately rounds to
    # zero in a Matern tail or a max-value CDF. A division by zero or a NaN is
    # not, and those are the ones this module is built to avoid, so raise on them.
    np.seterr(divide="raise", invalid="raise", over="raise", under="ignore")
    for test in (test_leave_one_out_moments, test_leave_one_out_candidate_moments,
                 test_permutation_invariance, test_truncated_variance,
                 test_conditional_variance_monte_carlo, test_gumbel_sampler, test_time_grid,
                 test_relevance_behaviour, test_importance_weights, test_cost):
        test()

    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("Failed: " + ", ".join(FAILED))
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
