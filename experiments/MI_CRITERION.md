# The mutual-information removal criterion

An alternative to W-DBO's Wasserstein criterion for deciding which stored
observations a Dynamic BO loop should forget. Implemented in
[`../src/wdbo_algo/mi_criterion.py`](../src/wdbo_algo/mi_criterion.py) (the
estimator) and
[`../src/wdbo_algo/mi_optimizer.py`](../src/wdbo_algo/mi_optimizer.py) (the
removal loop), selected with `--criterion mi`.

W-DBO scores an observation by how far the *whole* posterior moves when you drop
it. This scores it by how much it tells you about the thing the optimizer is
actually chasing — the instantaneous maximum at future times:

```
R(i) = ∫ ω(t) · I(y_i ; f*_t | D̃_i) dt  over t from t0 to ∞,
       ω(t) ∝ k_T(|t − t0|),  ∫ ω = 1
```

with `D̃_i = D \ {(x_i, t_i, y_i)}` and `f*_t = max_x f(x, t)`.

---

## 1. Reducing it to one number per observation

### The bridge variable

`y_i` is a past observation and `f*_t` is a future maximum, so unlike MES there
is no ready-made truncation `y_i ≤ f*_t` to exploit. Introduce `v = f(x_i, t)`
— the latent function at observation `i`'s own location, at the future time `t`.
Then:

- **(P1)** `y_i | v` is Gaussian, by the GP.
- **(P2)** `v | f*_t` is a Gaussian truncated above at `f*_t`.
- **(P3)** `y_i` and `f*_t` are independent given `v`. Once you know the function
  value at that place in the future, a past measurement at the same place says
  nothing more about the future maximum.

### The closed form

Write, all conditioned on `D̃_i`: `σ_y² = Var[y_i]` (noise included),
`σ_v² = Var[v]`, `c = Cov(y_i, v)`, `m_v = E[v]`, and `ρ² = c²/(σ_y²·σ_v²)`.
The law of total variance over `v`, using (P3) and the fact that `Var[y_i | v]`
does not depend on `v`, gives

```
s_i²(f̂*) = Var[y_i | f̂*_t, D̃_i] = σ_y² · [ 1 − ρ² · Δ(γ) ],   γ = (f̂*_t − m_v)/σ_v
```

where `Δ(γ) = 1 − Var[v | v ≤ f̂*]/σ_v²` is the **variance deficit** of a
standard normal truncated above at `γ`. Since `H(y_i | D̃_i) = ½·log(2πe·σ_y²)`
exactly, and OPES bounds `H(y_i | f̂*, D̃_i) ≤ ½·log(2πe·s_i²)`, the `σ_y²`
cancels and leaves

```
I(y_i ; f*_t | D̃_i) ≈ −½ · mean over f̂* in S_t of log[ 1 − ρ_i² · Δ(γ) ]    (★)
```

This is OPES's `log σ − mean log s`. Two properties worth noting:

- **(★) is ≥ 0 by construction**, and is a *lower* bound on the true mutual
  information — we upper-bounded the entropy being subtracted. For a removal
  rule that is the safe direction: it can understate but never overstate an
  observation's importance.
- Only `ρ²` and the truncation severity `γ` survive. The criterion asks how
  strongly an observation is correlated with the future function value at its
  own location, and how much that value is constrained by lying below the
  maximum.

## 2. All n leave-one-out posteriors from one matrix product

The literal reading of `D̃_i` is `n` refits per future time. It is not needed.
With `A = (K + σ_n²·I)⁻¹` over the **full** dataset, `â = A·(y − μ_const)`, and
`C_t[j, i] = k(z_j, (x_i, t))`, set `M_t = A·C_t` and `w = diag(M_t)`. Then for
every `i` simultaneously, **exactly**:

| quantity | closed form |
|---|---|
| `Var[y_i \| D̃_i]` | `1 / A_ii` (the standard GP LOO variance, R&W eq. 5.12) |
| `Cov(y_i, v_i \| D̃_i)` | `w_i / A_ii` |
| `Var[v_i \| D̃_i]` | `λ − Σ_j C_t[j,i]·M_t[j,i] + w_i²/A_ii` |
| `E[v_i \| D̃_i]` | `μ_const + (C_tᵀ·â)_i − w_i·â_i/A_ii` |

The `1/A_ii` terms are the rank-one correction that *undoes* conditioning on
observation `i`; drop them and these are the ordinary full-data moments. All
four are checked against explicit refitting in
[`validate_mi_criterion.py`](validate_mi_criterion.py) — agreement to 4e-16.

Separability makes even the remaining work cheap. Since
`k((x,t),(x',t')) = λ·k_S(x,x')·k_T(t,t')` and every point the criterion
evaluates sits at a spatial location that does not depend on `t` — either an
observation's own `x`, or a fixed candidate — we have
`C_t = λ·diag(k_T(t_j, t))·K_S`. The spatial Grams are built **once per clean**
and each future time is a row rescaling of them. Measured cost is ~130 ms and
roughly **flat in n past n ≈ 100**: it is dominated by the Gumbel quantile
search over the candidate set, not by the dataset.

## 3. Sampling f*_t

The MES recipe. Discretize the space with `--mi-candidates` Sobol points, treat
their values as independent so `P(f* ≤ z) ≈ Π_m Φ((z − μ_m)/σ_m)`, read the
25/50/75 percentiles off that CDF by bisection, fit a Gumbel by percentile
matching, and inverse-sample.

Sampled **per leave-one-out posterior by default**: `I(y_i ; f*_t | D̃_i)`
conditions on `D̃_i`, so the `f̂*` set that estimates it should come from `D̃_i`
too, not from a posterior that still contains the observation being scored. The
candidate moments under `D̃_i` are the same rank-one correction as §2 applied to
the candidate set,

```
E[f_m | D̃_i]   = μ_m − M_im · â_i / A_ii
Var[f_m | D̃_i] = σ²_m + M_im² / A_ii,     M = A·C_M
```

so forming all `n` of them costs one matrix product. What is *not* free is the
max-value solve on top: `n·|T|` Gumbel fits per clean instead of `|T|`. Measured
at `--mi-candidates 512`, `--mi-times 8`: 0.03 s → 0.6 s at `n = 50`, 0.11 s →
5.0 s at `n = 250`. Under a fixed wall-clock budget that is fewer queries, so
`--mi-fstar full` restores the shared full-`D` sample set (Week 2 §3.1's
argument: the correction is rank-one in a posterior conditioned on `n` points)
for runs where the cost matters more than the conditioning. `--mi-fstar is`
keeps that shared set and reweights it per `i` instead — see
[§3a](#3a-importance-sampling---mi-fstar-is-week-3).

| `--mi-fstar` | samples of `f*_t` | Gumbel fits per clean | `n = 250` cost |
|---|---|---|---|
| `loo` (default) | one set per `D̃_i` | `n·\|T\|` | 3–5 s |
| `full` | one set under `D`, shared | `\|T\|` | 0.05–0.11 s |
| `is` | the `full` set, reweighted towards each `D̃_i` | `\|T\|` | 0.07–0.15 s |

(`--mi-fstar-full` still works as an alias for `--mi-fstar full`.)

Implementation notes:

- The per-`i` percentile solves are **batched**: one active set of `n·3` scalar
  roots solved by safeguarded Newton, roots dropping out as they converge. A
  Python loop over `brentq` costs the same as everything else in the criterion
  put together at these sizes.
- The Gumbel inverse-sampling uses **common random numbers across `i`** — one
  set of uniforms, `n` different Gumbel parameters. Each `f̂*` row is still a
  correct marginal draw; sharing the uniforms cancels Monte-Carlo noise out of
  the comparison the removal loop actually makes, which is `argmin_i R(i)`, and
  makes the result independent of the order the dataset is stored in.
- Use the **latent** posterior, not `SpaceTimeGPModel.posterior()`, which wraps
  in `self.likelihood(...)` and so adds `σ_n²`. `f*` is a maximum of `f`, not
  of `y`.
- Bisect on `log P`, not `P`. A product of ~500 normal CDFs underflows to a flat
  zero well inside the textbook bracket, leaving the root-finder nothing to
  descend.

## 3a. Importance sampling (`--mi-fstar is`, Week 3)

### The idea

Estimate all `n` expectations `E_{f*_t | D̃_i}[H(y_i | f*_t, D̃_i)]` from the
**one** sample set `S_t ~ f*_t | D` that `full` already draws, by weighting it
differently for each `i`. By Bayes, with `D = D̃_i ∪ {y_i^obs}`,

```
p(f*_t | D̃_i) / p(f*_t | D) = p(y_i^obs | D̃_i) / p(y_i^obs | f*_t, D̃_i)      (exact)
```

and under the two assumptions the entropy step already makes — conditioning on
`f*_t` ≈ conditioning on `{v_i ≤ f*_t}` (c), and `y_i ⊥ f*_t | v_i` (d) — the
denominator is `p(y_i^obs | D̃_i) · Φ(γ_i^full)/Φ(γ_i^loo)`, so

```
w_i(f̂*) = Φ(γ_i^loo) / Φ(γ_i^full) = P(v_i ≤ f̂* | D̃_i) / P(v_i ≤ f̂* | D)
γ_i^loo  = (f̂* − E[v_i | D̃_i]) / sd[v_i | D̃_i],   γ_i^full likewise under D

E[H(y_i | f*_t, D̃_i)] ≈ Σ_{f̂* ∈ S_t} w̄_i(f̂*) · H(y_i | f̂*, D̃_i),   w̄_i = w_i / Σ w_i
```

Self-normalized because (c) and (d) are approximations, so the raw weights do
not average to one. `v_i | D` needs no rank-one correction (`full_data_moments`);
`v_i | D̃_i` is already computed for the entropy term. The ratio is formed as
`exp(log_ndtr − log_ndtr)`: for a sample far below `E[v_i]` both `Φ`s underflow.
Cost is `full` plus one `n × n` product per future time — 0.07–0.15 s against
`loo`'s 3–5 s at `n = 250` (timings vary run to run).

### What it does in practice: the weights are ≈ 1

Measured by [`compare_fstar_sampling.py`](compare_fstar_sampling.py), on
GP-prior datasets with the hyperparameters the surrogate actually fits
(medians from logged `ackley4d` / `temperature` runs, plus a low-noise 2-D case):
the effective sample size of the weights is **≥ 0.9999·K for every (i, t)** in
all three scenarios, so `is` reproduces `full` to within Monte-Carlo noise.

The reason is structural, not numerical. `w_i` moves away from 1 only when
`P(v_i ≤ f̂*)` differs between `D` and `D̃_i`, i.e. only when an observation's
*own* future value is within a few standard deviations of the maximum. For
everything else `Φ(γ^loo) ≈ Φ(γ^full) ≈ 1`. But the leave-one-out shift of
`f*_t` is real for such points too — dropping `y_i` inflates the posterior
variance of the whole neighbourhood of `x_i`, which moves the max — and
assumption (c) is exactly what discards that channel: it keeps only the bound
`v_i ≤ f*_t` and throws away "a high maximum makes the function near `x_i`
higher too". The weights are therefore *consistent with the model's own
entropy approximation* (under (c), `y_i` carries almost no information about
`f*_t` unless `v_i` is near it, so its removal should barely move `f*_t`
either), but they cannot recover the LOO shift that `loo` samples.

### How much any of this matters: the three modes against references

Two references per scenario, candidate set fixed so only the sampling differs:
**joint-LOO** samples `f*_t | D̃_i` from *joint* (correlated) posterior draws
over the candidates — no independence approximation, the covariances for all
`i` via `Cov_i = Cov + m_i m_iᵀ / A_ii` — and **Gumbel-LOO** is `loo` at
`K = 4000`, i.e. what `loo` converges to. 12 replicates per `(mode, K)`.
Error is the RMSE of `log10(estimate / reference)` over observations.

| scenario | mode | vs Gumbel-LOO, K=32 | K=128 | vs joint-LOO, K=128 | argmin = joint-LOO's |
|---|---|---|---|---|---|
| ackley-like (d=4, n=60, σ²=0.17) | full | 0.046 | 0.025 | 0.219 | 100 % |
| | loo | 0.045 | 0.024 | 0.218 | 100 % |
| | is | 0.046 | 0.025 | 0.219 | 100 % |
| temperature-like (d=3, n=45, σ²=0.6) | full | 0.048 | 0.028 | 0.796 | 100 % |
| | loo | 0.046 | 0.024 | 0.797 | 100 % |
| | is | 0.048 | 0.028 | 0.796 | 100 % |
| low noise (d=2, n=60, σ²=0.02) | full | 0.058 | 0.037 | 1.276 | 83 % |
| | loo | 0.051 | 0.026 | 1.281 | 83 % |
| | is | 0.058 | 0.037 | 1.276 | 83 % |
| *all three* | joint-full (no LOO, exact max) | | | 0.007 – 0.011 | 100 % |

Read in that order:

1. **Against what `loo` converges to**, the three modes are within Monte-Carlo
   noise of each other; `full`/`is` keep a bias of ~0.01 decades (≈ 2–5 %) that
   `loo` does not, visible only in the low-noise case. Spearman ≥ 0.999
   everywhere.
2. **Against the faithful reference, all three are off by 0.2–1.3 decades,
   identically.** Joint samples *without* leave-one-out (`joint-full`) are off
   by only 0.007–0.011. So the error in `f*_t` sampling comes almost entirely
   from the **independence approximation in the Gumbel CDF**, not from which
   posterior it is conditioned on. Treating hundreds of strongly correlated
   candidates as independent overstates `f*_t` (in the 2-D case, median 1.7
   vs 0.9), which pushes every `γ` up and every relevance down — by more when
   `l_S` is long relative to the candidate spacing.
3. In the two realistic scenarios the error is close to a **uniform shift in
   log space** (Spearman 0.997–0.9998 vs joint-LOO, argmin always right), so
   for *ranking* — the `argmin` the removal loop takes — it mostly washes out.
   In the low-noise case it is not uniform (−0.5 to −2 decades per point,
   Spearman 0.97) and the argmin is right in only 83 % of replicates — equally
   for all three modes. It
   does not wash out of the **budget**: `--mi-alpha` is calibrated in nats
   against the biased scale.

### Recommendation

- `is` is correct as derived and costs as little as `full`, but at the
  hyperparameters we fit it is numerically indistinguishable from `full`.
  It only departs from `full` for observations whose future value sits near the
  maximum, which are never removal candidates.
- `loo` buys ~0.01 decades over `full` at 20–40× the cost. At a fixed
  wall-clock budget, `full` or `is` is the better trade.
- The error worth fixing is the Gumbel independence approximation (item 2), and
  it is shared by all three modes.

Figures: `data/fstar_sampling/` — `per_point_bias.png` (error per observation vs
joint-LOO), `error_vs_samples.png`, `fstar_distributions.png` (the `f*_t` CDFs
of all five samplers for the observation with the largest true LOO shift),
`runtime.png`, and `summary.json`.

## 4. Time grid and weights

`T = {t0 + l_T·u_j}` on Gauss-Legendre nodes over `[0, H]`, default `H = 3·l_T`
(`--mi-horizon`), with weights `ω_j ∝ gl_weight_j · k_T(t_j − t0)` normalized to
sum to 1. `--mi-times` sets the node count (default 8).

Week 2 flags a **double decay**: `I(y_i; f*_t)` already vanishes as `t` moves
away, and `ω(t) ∝ k_T` applies the same decay again. `--mi-weight uniform` drops
the kernel factor. Measured on two real datasets across six lengthscales:
kernel weighting is a **uniform factor of 1.6–2.6 larger**, with rank
correlation **0.997–0.9999** against uniform. It is a constant that a change in
α absorbs completely, and it essentially never changes *which* observation is
least relevant — so the choice does not matter much either way.

## 5. Numerics

Three traps, all silent rather than loud. A wrong criterion does not crash; it
deletes the wrong observations and reports a plausible regret curve.

- **`φ(γ)/Φ(γ)` is 0/0 below γ ≈ −38.** Evaluate it as `exp(log φ − log Φ)` with
  `scipy.special.log_ndtr`.
- **The textbook `Var = 1 − γ·λ − λ²` cancels catastrophically as γ → −∞**: both
  terms are `O(γ²)` while the result decays to `1/γ²`. Below γ = −30 use the
  asymptotic `1/γ² − 6/γ⁴`.
- **The deficit `Δ(γ)` must be computed directly, never as `1 − Ψ`.** As
  γ → +∞, `Δ → 0` — and that is precisely the regime the removal loop lives in,
  a stale observation being one whose future value is nowhere near the maximum.
  `γ·λ + λ²` is a sum of two positives and stays exact to 1e-300; recovering it
  as `1 − Ψ` rounds to zero below 1e-16. That matters, because relevance values
  which underflow to exactly 0.0 all **tie**, and `argmin` then picks among them
  arbitrarily — the ranking silently stops working. For the same reason (★) is
  evaluated with `log1p`.

## 6. The removal budget

W-DBO's criterion is a Wasserstein distance normalized against the prior, so it
is dimensionless and bounded in `[0, 1]`, and its budget is multiplicative.
`mi_criterion` returns **nats**, which are additive. So:

```
b += α · Δt / l_T     accrue α nats of discardable information per lengthscale
b -= R(i)             consume on removal
b starts at 0.0       nothing is removable before any time has passed
```

`--mi-budget-cap` clamps `b`, so a clean that removes nothing cannot bank budget
and purge a burst later. `--min-dataset-size` (default 15, the initial-design
size) is a floor the loop never removes below.

### These are the same rule as W-DBO's, in different units

Take logs of W-DBO's own loop: `b *= (1+α)^(Δt/l_T)` is
`log b += log(1+α)·Δt/l_T`, and `b /= min_crit` is `log b -= log(min_crit)`. So
W-DBO is *already* an additive budget, in units of `log(1 + criterion)`, and its
**α = 1/4 means "discard ln(1.25) = 0.223 nats per temporal lengthscale"**.

Setting `criterion := exp(R(i)) − 1` would therefore drop this criterion into
W-DBO's machinery unchanged. It is not worth doing: `e^R ≈ 1 + R`, so for the
tiny values this criterion produces the exponential is an affine shift that
destroys the ranking. Measured: 20.4 decades of spread collapse onto
`[1.0000000000000000, 1.0015150647222844]`, with the minimum exactly 1.0.

### Why α is not transferable

W-DBO's 0.223 nats/lengthscale works because its criterion is normalized into
`[0, 1]`, so every removal costs `log(1+R)` in `[0, 0.69]` nats. `mi_criterion`
is **not normalized**, and on `ackley4d` its scores span roughly 1e-24 to 1e-3
nats. That is why `--mi-alpha` defaults to 1e-11 while `--alpha` defaults to
0.25, as separate flags: the two numbers are not comparable, and conflating them
silently produces either an arm that removes nothing or one that empties to the
floor.

## 7. The lengthscale dominates everything

`R(i) ≈ ½·ρ_i²·Δ`, and `ρ_i` carries a factor `k_T(Δt)`, so `R` carries
`k_T(Δt)²`. For Matérn-3/2 that is `exp(−2·√3·Δt/l_T)`. **`l_T` appears only as
a divisor of `Δt` inside an exponential**, which has three consequences.

1. The criterion measures age in *lengthscales*, not in clock time. The same
   observation is "0.1 lengthscales old" at `l_T = 5` and "6 lengthscales old"
   at `l_T = 0.08`.
2. Each lengthscale of age costs a factor of 50–100. Measured slope on two
   independent real datasets: **−1.7 to −2.0 decades per lengthscale**, against
   the −1.504 that `k_T²` alone predicts; leave-one-out conditioning supplies
   the rest. (A *decade* here means a factor of 10.)
3. The dataset spans ~1 clock unit, so the oldest point's age is about `1/l_T`
   lengthscales and `log10 R_min ≈ −1.9/l_T + const` — a reciprocal *inside* the
   exponent:

   | `l_T` | oldest point's age | `R_min` |
   |---|---|---|
   | 5.25 | 1.2 lengthscales | 5e-8 |
   | 0.233 | 4.3 | ~1e-12 |
   | 0.08 | 13.5 | 1e-28 … 1e-33 |

Other hyperparameters (`l_S`, `λ`, `σ_n`) shift the *level* by a few decades
through redundancy — how much the other `n−1` observations already explain. Only
`l_T` changes the *slope*.

**The consequence for calibration.** `b += α·Δt/l_T` is linear in `1/l_T`; the
criterion is exponential in it. A fixed α in nats is therefore not a fixed
staleness threshold — it drifts by roughly 80× for every lengthscale the MLE
shaves off `l_T`. Since `l_T` is re-estimated every iteration and ranges over
0.087–5.25 across seeds of the *same* benchmark, one α can mean "remove nothing"
on some seeds and "remove down to the floor" on others. This is the open problem
with the criterion as specified, and the argument for normalizing it.

It is also why `queries.csv` carries **two** lengthscale columns. `lT` is
post-cleaning — what the next iteration acts on — while `criterion_lT` is the
one the score was actually computed under. Every removal refits the
hyperparameters, so within a single clean the two can differ by orders of
magnitude, and calibration needs a score paired with the lengthscale that
produced it.

## 8. Using it

```bash
# The MI arm
python experiments/synthetic/run_experiment.py --criterion mi --mi-alpha 1e-11 --n-seeds 10

# W-DBO baseline, same floor, same clock
python experiments/synthetic/run_experiment.py --criterion wasserstein --alpha 0.25 --n-seeds 10

# No-removal ablation
python experiments/synthetic/run_experiment.py --criterion none --n-seeds 10

# Calibrate. alpha = 0 never removes but still pays the criterion's cost, so
# min_criterion in queries.csv gives the nat scale for free.
python experiments/sweep_mi_alpha.py experiments/synthetic/run_experiment.py \
    --alphas 0 1e-12 1e-11 1e-10 1e-9 --n-seeds 3 --duration-seconds 150
```

Sweep in **decades** — the criterion is log-distributed, so linear steps tell
you nothing.

`--min-dataset-size` defaults to 15 for all arms, including W-DBO. Pass `2` to
reproduce the original W-DBO behaviour; the library default in
`WDBOOptimizer.__init__` is still 2, so the vendored algorithm is unchanged.

## 9. Validation

```bash
python experiments/validate_mi_criterion.py
```

57 checks, none of which need an experiment to run: the four leave-one-out
identities against explicit refitting (4e-16), `s_i²` against a 2M-draw Monte
Carlo over the truncated bridge variable (0.05–0.23 %), the deficit against
`scipy.stats.truncnorm` and against both its asymptotes, the Gumbel samples
against the quartiles of the max-value CDF they were fitted to, and end-to-end
properties of the relevance vector — non-negative, permutation-equivariant,
lower for stale observations, near zero for a duplicated one, and free of ties
in the small-`l_T` regime where it previously underflowed. For `--mi-fstar is`:
the full-data moments of `v_i` against the GP formula, the Bayes step that the
importance weights rest on (conditioning `(y_i, v_i) | D̃_i` on the observed
`y_i` recovers `v_i | D`, Week 3 eq. 15, to 2e-16), the weights against the
direct `Φ` ratio, and their finiteness when both `Φ`s underflow.

How close each sampling mode gets to the quantity it estimates is a statistical
question, not an identity, so it lives in a separate script — see §3a:

```bash
python experiments/compare_fstar_sampling.py [--quick]
```

## 10. References

- Wang & Jegelka, *Max-value Entropy Search for Efficient Bayesian
  Optimization*, ICML 2017 — the `f*` sampling.
- Hoffman & Ghahramani, *Output-Space Predictive Entropy Search*, NeurIPS
  workshop 2015 — the Gaussian upper bound on the conditional entropy.
- Bardou, Thiran & Ranieri, *This Too Shall Pass: Removing Stale Observations in
  Dynamic Bayesian Optimization*, 2024 — the loop, the space-time GP, and the
  criterion this one is an alternative to.
