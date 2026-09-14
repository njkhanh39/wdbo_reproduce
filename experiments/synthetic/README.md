# Reproducing the WDBO synthetic-function benchmarks

This reproduces the synthetic half of Appendix H.2 of the WDBO paper ("This
Too Shall Pass: Removing Stale Observations in Dynamic Bayesian
Optimization", Bardou, Thiran & Ranieri, 2024,
https://arxiv.org/abs/2405.14540) — the ten analytic benchmarks in Table 2
and Figures 8–19 (Rastrigin, Schwefel, Styblinski-Tang, Eggholder, Ackley,
Rosenbrock, Shekel, Hartmann-3, Hartmann-6, Powell).

It is the analytic sibling of [`../temperature/`](../temperature/): the
optimizer, the real-time loop, the seed handling, the CSV/plot outputs, and
the ±1 std bands are all identical. **Only the objective differs** — a
closed-form function instead of an interpolated real-world surface — so this
page focuses on that difference and defers to
[`../temperature/README.md`](../temperature/README.md) §4–6 for everything
shared (what `--seed` controls, the metric definitions, how to read the
uncertainty bands and the 200-row CSV).

**Currently implemented: `ackley4d`.** Adding the others is a few lines each
— see §5.

## 1. How a static test function becomes a dynamic objective

The paper writes every synthetic benchmark as `f(z)` with
`z = (x_1, …, x_d, t)` and `d' = d + 1` (H.2). In other words **the temporal
axis is just the last input of an ordinary optimization test function**, and
the spatial optimum drifts as the clock advances.

Two scale conventions matter, both from H.1:

- **The `d` spatial axes use the function's natural box** (Ackley:
  `[−32, 32]^d`). `ackley4d` is `d' = 4`, so the optimizer works in a **3-D**
  spatial box `[−32, 32]^3` with time as the 4th axis. (`temperature` is
  fixed at `d' = 3`, `[0, 1]^2` spatial.)
- **The temporal axis is normalized to `[0, 1]` — not rescaled to the
  spatial box.** H.1 says "the temporal domain [...] is normalized in
  `[0, 1]`", and the paper's Table 5 temporal lengthscales (`l_T ≈ 0.6`
  "axis unit") only make sense on a `[0, 1]` axis. Rescaling time into
  `[−32, 32]` (an over-literal reading of "optimized on `[lo, hi]^d'`")
  would make Ackley's `cos(2πt)` term flip sign 64× across the run, leaving
  successive-in-time observations uncorrelated — W-DBO would then correctly
  purge its whole dataset every step, contradicting the moderate dataset
  sizes it shows in Figure 13. `src/test.py`'s toy demo *does* rescale its
  clock into the Rosenbrock box (`5t/8 − 1`); the real benchmarks do not.

  For Ackley specifically the spatial argmin stays at the origin for every
  `t` regardless (both terms are minimized there), so the "dynamic" part is
  a smooth single-period ripple in the *achievable value*, not a moving
  optimum — the same is true under either scale convention; §5.2 of the
  paper frames Ackley's difficulty as an exploration/acquisition problem,
  not an optimum-tracking one.

**Sign flip.** These functions are *minimized* (Ackley's global optimum is
its deep central well), but W-DBO *maximizes* its acquisition — the task
analogue of "activate the hottest sensor". So `objective.py` hands the DBO
loop `−f`, and `oracle(t) = max_x (−f(x, t)) = −min_x f(x, t)`. Regret =
`oracle(t) − (−f(query))` = `f(query) − min_x f`, i.e. non-negative and in
the function's own units (comparable to the paper's Table 2).

## 2. Pipeline

No preprocessing and no data download — the objective is analytic.

```
benchmarks.py    →  Benchmark: a vectorized f(z), its d spatial box,
       │              minimize flag  (time axis is always [0, 1])
       ▼
objective.py     →  f(x, t)  (append clock as last axis + sign flip)
       │              + oracle(t) = max_x (−f(x, t)) via a dense spatial
       │                grid search, cached to
       │                data/synthetic/<benchmark>/oracle.npz
       ▼
run_experiment.py → runs WDBOOptimizer against the objective,
                     logs regret & dataset size, saves CSVs + plots to
                     data/synthetic/<benchmark>/results/
```

### The oracle

`oracle(t)` is built once by `compute_oracle_curve`: a regular
`grid_resolution ** spatial_dim` spatial grid, evaluated at
`oracle_time_points` times in `[0, 1]`, taking the max of `−f` at each time;
the curve is cached and read with `np.interp` in the hot loop. Notes:

- **`--oracle-grid-resolution` (default 33, odd).** Odd so a symmetric-domain
  optimum lands exactly on a node — for Ackley the spatial argmin is the
  origin at every `t`, so an odd grid makes the oracle *exact* (verified: 0
  regret for origin queries, no negative regrets over 50k random probes).
  The grid has `33^3 ≈ 3.6·10^4` nodes for `ackley4d`; this is only
  practical while `spatial_dim ≲ 3`.
- **`--oracle-time-points` (default 1000).** With time in `[0, 1]`, Ackley's
  `oracle(t)` is a smooth single-period ripple (range ≈ `[−2.16, 0]`), so
  `np.interp` on 1000 samples is far more than enough. First run spends
  ~3 s here, then it is cached.
- **Noise.** `estimate_noise_std` sets `σ` so `Var(noise) = 5 % ·` signal
  variance (paper H.1), estimating the signal variance by sampling `f`
  uniformly over the full `d'` box, time included (deterministic; ≈ 0.38 for
  `ackley4d`).

## 3. Running it

```bash
python experiments/synthetic/run_experiment.py --benchmark ackley4d --n-seeds 10
```

(Needs the same environment as the temperature experiment — the WSL venv with
the compiled `wdbo_criterion` extension; see [`../../NOTE.md`](../../NOTE.md).)

| flag | default | paper value | notes |
|---|---|---|---|
| `--benchmark` | `ackley4d` | — | key in `benchmarks.py` |
| `--duration-seconds` | 600 | 600 (10 min) | real wall-clock budget per replication |
| `--n-initial-observations` | 15 | 15 | |
| `--alpha` | `1/3` | `1/3` | removal-budget hyperparameter (§5.1 sensitivity analysis / Table 2). Note `temperature`'s script defaults to 0.25. |
| `--n-seeds` | 10 | 10 | independent replications to average |
| `--seed` / `--same-seed` | 0 / off | — | see [`../temperature/README.md`](../temperature/README.md) §4 |
| `--oracle-time-points` | 1000 | — | oracle curve resolution (§2) |
| `--oracle-grid-resolution` | 33 | — | oracle spatial grid, per axis (§2) |

Kernels and the real-time loop are identical to `temperature`: Matérn-5/2
spatial, Matérn-3/2 temporal, clock `= elapsed / duration_seconds`.

A full `--n-seeds 10 --duration-seconds 600` run is ~100 min of wall time
(10 × 10 min); use `--n-seeds 1 --duration-seconds 60` for a smoke test.

## 4. Reading the results

Written to `data/synthetic/<benchmark>/results/` — same four files, same
meanings, as the temperature experiment. See
[`../temperature/README.md`](../temperature/README.md) §5–6 for the full
description; in brief:

- **`regret.csv`** — with `--n-seeds > 1`, exactly 200 rows on a fixed
  `linspace(0, 600, 200)` wall-clock grid (a resampling, *not* the query
  count), holding the seed-averaged **instantaneous** regret and dataset
  size.
- **`summary.csv`** — the one-number headline: mean and variance across
  replications of each replication's own average regret and average response
  time. For `ackley4d` the paper's Table 2 reference is W-DBO average regret
  ≈ **2.24**.
- **`regret_and_size_vs_duration.png`** — left: **running** average regret vs
  duration (±1 std band); right: dataset size vs duration on a log axis. The
  size band can visually drop to the axis floor whenever `std ≥ mean` — a
  clip artifact, not a real collapse (temperature README §6).
- **`regret_vs_response_time.png`** — every query scattered, plus the
  mean ±1 std marker; the single-algorithm analogue of the paper's
  per-benchmark left panel.

## 5. Adding another benchmark

1. Add a vectorized `func(z: (n, d') array) -> (n,) array` and a `Benchmark`
   entry to [`benchmarks.py`](benchmarks.py), then list it in `BENCHMARKS`.
   Use the paper's H.2 constants and domain (e.g. Rastrigin: `a = 10`,
   `d' = 5`, `[−4, 4]^5`).
2. Run `--benchmark <name>`. The oracle is computed and cached automatically.

**Caveat — the grid oracle only scales to `spatial_dim ≲ 3`.** `33^d`
explodes past that (Rastrigin5d → `33^4 ≈ 1.2·10^6` nodes × time points;
Hartmann-6 → `33^5`). Those benchmarks need an optimizer-based oracle
(multi-start L-BFGS per time slice) instead of `compute_oracle_curve` — not
yet implemented here.

## 6. What matches / approximates / is out of scope

- **Matches**: the benchmark definitions and constants (H.2), the
  spatio-temporal lifting (`z = (x, t)` on one shared box), kernels, initial
  observation count, `alpha = 1/3`, wall-clock budget, 10-replication
  averaging, and the 5 %-signal-variance noise model (H.1).
- **Approximates**: `oracle(t)` — a dense grid search (the paper does not
  state how it computes its oracle). For `ackley4d` it is effectively exact
  (the spatial optimum sits on a grid node at every `t`); for a benchmark
  whose optimum falls between nodes it is a grid-resolution approximation.
- **Out of scope**: the baselines (GP-UCB, TV-GP-UCB, ABO, ET-GP-UCB,
  R-GP-UCB) — this reproduces W-DBO's own curve, not the comparison in
  Table 2 / Figure 13; and the ARD-SE-kernel variants (paper Figures 10, 14).
