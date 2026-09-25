# Reproducing the WDBO synthetic-function benchmarks

This reproduces the synthetic half of Appendix H.2 of the WDBO paper ("This
Too Shall Pass: Removing Stale Observations in Dynamic Bayesian
Optimization", Bardou, Thiran & Ranieri, 2024,
https://arxiv.org/abs/2405.14540) — the ten analytic benchmarks in Table 2
and Figures 8–19 (Rastrigin, Schwefel, Styblinski-Tang, Eggholder, Ackley,
Rosenbrock, Shekel, Hartmann-3, Hartmann-6, Powell).

It is the analytic sibling of [`../temperature/`](../temperature/): the
optimizer, the real-time loop, the seed handling, the CSV/plot outputs, and
the plotting conventions are all identical. **Only the objective differs** — a
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

Two scale conventions matter:

- **The `d` spatial axes use the function's natural box** (Ackley:
  `[−32, 32]^d`). `ackley4d` is `d' = 4`, so the optimizer works in a **3-D**
  spatial box `[−32, 32]^3` with time as the 4th axis. (`temperature` is
  fixed at `d' = 3`, `[0, 1]^2` spatial.)
- **The temporal axis spans the same box as the spatial ones.** H.2 gives
  one domain per benchmark covering all `d'` axes — "we optimized the
  function on the domain `[−32, 32]^d'`" with `d' = 4` and
  `z = (x_1, …, x_d, t)` — so Ackley's time argument runs over `[−32, 32]`,
  like its three spatial arguments. H.1's "the temporal domain [...] is
  normalized in `[0, 1]`" describes a convenience of the reference
  implementation, not the function. The loop here runs on **absolute
  environment time** — see [`../README.md`](../README.md) §2 — so this span is
  the axis the experiment really moves along, and its width divided by 600 s
  is the environment speed the paper's setting implies.

  Each benchmark carries its own `env_span` in `benchmarks.py`;
  `--env-span 0 1` runs the other reading (time literally in `[0, 1]`
  while space spans `[−32, 32]`, which makes Ackley far smoother in time
  than in space) as a sensitivity check. Note it changes the default
  `--env-speed` too, since that is derived from the span's width. Saved
  results under `results_t-32_32/` predate the environment clock.

  For Ackley the spatial argmin stays at the origin for every `t` under
  either convention (both terms are minimized there), so the "dynamic" part
  is a ripple in the *achievable value*, not a moving optimum; §5.2 of the
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
       │              its env_span, minimize flag
       ▼
objective.py     →  f(x, t_env)  (append env clock as last axis + sign flip)
       │              + oracle(t_env) = max_x (−f(x, t_env)) via a dense
       │                spatial grid search over ABSOLUTE env times, cached
       │                to data/synthetic/<benchmark>/
       │                     oracle_t<lo>_<hi>_d<density>_g<grid>.npz
       ▼
run_experiment.py → runs WDBOOptimizer against the objective,
                     logs regret & dataset size, saves CSVs + plots to
                     data/synthetic/<benchmark>/results/
```

### The oracle

`oracle(t)` is built once by `compute_oracle_curve`: a regular
`grid_resolution ** spatial_dim` spatial grid, evaluated at
absolute environment times covering `env_span` at `--oracle-density` samples
per unit of time, taking the max of `−f` at each time; the curve is cached and
read with `np.interp` in the hot loop. Notes:

- **`--oracle-grid-resolution` (default 33, odd).** Odd so a symmetric-domain
  optimum lands exactly on a node — for Ackley the spatial argmin is the
  origin at every `t`, so an odd grid makes the oracle *exact* (verified: 0
  regret for origin queries, no negative regrets over 50k random probes).
  The grid has `33^3 ≈ 3.6·10^4` nodes for `ackley4d`; this is only
  practical while `spatial_dim ≲ 3`.
- **`--oracle-density` (default 64, samples *per unit of environment time*).**
  Ackley's `cos(2πz)` term completes one period per unit of time, so this is
  ≈64 samples per period; over `[−32, 32]` that is ≈4097 samples for a curve
  oscillating 64× (range ≈ `[−20.2, −0.02]`). Under `--env-span 0 1` it is a
  single smooth ripple (range ≈ `[−2.16, 0]`) and far less is ample.

  The density is per *unit*, not per *run*, and that is the whole point: the
  old `--oracle-time-points` fixed the sample **count** per table, so changing
  the span silently changed the resolution per unit of time. An under-sampled
  `f*` misses peaks, which biases it **low**, which makes regret look
  **better** than it is — and it fails silently. Before trusting a new
  `--env-speed`, double this and check `f*` does not move. First run spends a
  few seconds here, then it is cached (per span, density and grid — see the
  filename tag).
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
| `--duration-seconds` | 600 | 600 (10 min) | real wall-clock budget per replication. Buys compute only — it no longer changes how fast the environment moves |
| `--n-initial-observations` | 15 | 15 | drawn over `S' ×` the first 1/40 of the reference run's environment interval, per H.1 |
| `--env-span` | benchmark's own | H.2 box (Ackley: `−32 32`) | the benchmark's whole temporal domain; see §1 |
| `--env-speed` | `span / 615` | — | environment units per real second. The default makes a 600 s run cover `--env-span` exactly, i.e. the paper's setting. See [`../README.md`](../README.md) §2 |
| `--env-t0` | low end of span | — | environment time the initial design starts at |
| `--alpha` | `0.25` | `1/4` | removal-budget hyperparameter. §5.1: "the sweet spot is reached for α = ¼. This hyperparameter value is used to evaluate W-DBO in the next section." |
| `--n-seeds` | 10 | 10 | independent replications to average |
| `--seed` / `--same-seed` | 0 / off | — | see [`../temperature/README.md`](../temperature/README.md) §4 |
| `--oracle-density` | 64 | — | oracle samples per unit of environment time (§2) |
| `--oracle-grid-resolution` | 33 | — | oracle spatial grid, per axis (§2) |
| `--label` | none | — | names the timestamped results directory, e.g. `--label paper` |
| `--results-dir` | none | — | write here instead of a fresh timestamped directory |

Kernels and the real-time loop are identical to `temperature`: Matérn-5/2
spatial, Matérn-3/2 temporal, environment clock
`t_env = env_start + env_speed × elapsed_seconds`. Per H.1 the 15 initial
observations are drawn uniformly over `S' ×` the first fortieth of the
reference run's environment interval — spread out, not stacked at one instant
— and the wall clock starts only once they are in hand, with their real cost
reported separately as `warmup_seconds` rather than assumed to be `H/40`.
Initial observations are not queries the algorithm chose, so they are
excluded from the regret log.

A full `--n-seeds 10 --duration-seconds 600` run is ~100 min of wall time
(10 × 10 min); use `--n-seeds 1 --duration-seconds 60` for a smoke test.

## 4. Reading the results

Written to a fresh timestamped directory
`data/synthetic/<benchmark>/results/<YYYYmmdd-HHMMSS>[-label]/` —
same files, same meanings, as the temperature experiment, since both share
[`../common.py`](../common.py). See
[`../temperature/README.md`](../temperature/README.md) §5–6 for the full
description; in brief:

- **`queries.csv`** — the raw per-query log, every seed, no resampling. The
  only irreplaceable file; everything else is a view over it. Includes
  `t_response` (our response time: acquisition + fit + clean, which departs
  from the paper), `t_acq_fit` (H.1's definition, *excluding* cleaning, kept
  for comparison with the paper), the `t_acq` / `t_fit` / `t_clean` parts and
  `t_eval`, plus `n_removed`, and the MLE hyperparameters
  `lambda, lS, lT, noise` plus `removal_budget` per query. `env_time` is
  absolute environment time, not a normalized fraction. Also the queried point
  `x_0..x_2`, its reading `y`, the noise-free `true_value`, and the stage
  timestamps (`t_apply` is when `x` took effect) — these are what let a run be
  re-scored afterwards. Each replication opens with an `iteration = −1` row
  holding the initial configuration; it is excluded from every metric.
- **`per_seed.csv`** — one row per replication: iteration count, integrated
  `time_avg_regret` and separate query mean `avg_regret`, mean response time (acquisition + fit + clean), H.1's acquisition + fit time and clean time,
  final/max/min dataset size, `median_lT`, `total_removed`, plus
  `warmup_seconds` and the environment interval the run covered.
- **`summary.csv`** — the quotable numbers as `mean, sem, n_runs`. For
  `ackley4d` the paper's Table 2 reference is average regret ≈ **2.24**; the
  script prints yours next to it.
- **`run.json`** — args, git commit, host, CPU, torch thread count, versions.
  The benchmark is wall-clock-driven, so the machine is an experimental
  parameter and two runs are only comparable if this matches.
- **`regret_and_size_vs_duration.png`** — left: **time-average regret up to t**
  (integrated from the configuration held on each interval, including changes
  between grid points); right: dataset size on a log axis. Both panels draw the across-seed
  mean with every seed faint behind it — seed outcomes can be bimodal, so the
  individual lines matter (temperature README §6).
- **`regret_vs_response_time.png`** — every query scattered against its
  acquisition + fit + clean time (`t_response`), plus the
  one marker per seed and the mean of those; the single-algorithm analogue of
  the paper's per-benchmark left panel.
- **`lengthscale_and_budget.png`** — not in the paper. `lT` and the removal
  budget per seed, both log-scaled. The budget grows as
  `(1 + alpha) ** (Δt / lT)`, so an `lT` driven towards zero explodes it and
  purges the dataset to the floor of 2; this panel is where that gets
  diagnosed.

Re-render any figure from a finished run without repeating it:

```bash
python experiments/plot.py data/synthetic/ackley4d/results/<run dir>
```

Runs under `data/synthetic/ackley4d/saved/` predate this layout entirely, so
`plot.py` cannot read them. Runs under `results_t-32_32/` predate the
environment clock: `plot.py` still renders them (`load_run` renames
`sim_time` → `env_time` and `t_response` → `t_acq_fit`), but their
environment speed was `span / duration_seconds` and they carry no timing
split, so their numbers are **not** comparable to new ones.

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
  spatio-temporal lifting (`z = (x, t)` on one shared box), kernels, the 15
  initial observations over `S' × [0, 1/40]`, `alpha = 1/4`, the 600 s
  wall-clock budget, 10-replication averaging, and the 5 %-signal-variance
  noise model (H.1).
- **Approximates**: `oracle(t)` — a dense grid search (the paper does not
  state how it computes its oracle). For `ackley4d` it is effectively exact
  (the spatial optimum sits on a grid node at every `t`); for a benchmark
  whose optimum falls between nodes it is a grid-resolution approximation.
- **Out of scope**: the baselines (GP-UCB, TV-GP-UCB, ABO, ET-GP-UCB,
  R-GP-UCB) — this reproduces W-DBO's own curve, not the comparison in
  Table 2 / Figure 13; and the ARD-SE-kernel variants (paper Figures 10, 14).
