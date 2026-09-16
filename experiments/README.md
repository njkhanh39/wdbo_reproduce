# Experiment Setup

Both experiments reproduce W-DBO — Wasserstein-based Dynamic Bayesian
Optimization, from Bardou, Thiran & Ranieri, "This Too Shall Pass: Removing
Stale Observations in Dynamic Bayesian Optimization" (2024) — against a
moving-target objective `f(x, t)`, following Appendix H.1–H.2. This page
covers only the experiment setup: what's shared between the two benchmarks,
then what's specific to each. See [`synthetic/README.md`](synthetic/README.md)
and [`temperature/README.md`](temperature/README.md) for full detail on
results, metrics, and known issues.

## Shared setup

- **Optimization loop, logging, and plotting** all live in
  [`common.py`](common.py) — the two benchmarks differ only in how they build
  their objective, so they cannot drift apart.
- **Kernels**: Matérn-5/2 over the spatial dimensions, Matérn-3/2 over time.
- **Budget**: each replication runs for a fixed 600s wall-clock duration
  (10 minutes, matching the paper); the optimizer's internal clock sweeps
  linearly over `[0, 1]` as `elapsed / duration`.
- **Initial design**: 15 observations drawn uniformly over `S' × [0, 1/40]`
  (H.1) — spread across the first fortieth of the horizon rather than stacked
  at `t = 0`, so the initial batch carries information about the temporal
  lengthscale. This window is charged against the 600s budget; the
  optimization loop proper starts at `t = 1/40`, and these points are excluded
  from the regret log since the algorithm didn't choose them.
- **Removal hyperparameter**: `alpha = 0.25`, the paper's reported "sweet
  spot" (§5.1) and the value used for every Table 2 number.
- **Noise**: observations are corrupted with Gaussian noise at 5% of the
  objective's signal variance (H.1).
- **Replication**: `--n-seeds 10` averages across independent runs, each with
  its own seed, noise draw, initial points, and acquisition-optimizer
  restarts (see each benchmark's README for exactly what the seed does and
  doesn't pin down).
- **Outputs**: a raw per-query CSV (the only irreplaceable artifact),
  per-seed and summary CSVs, `run.json` provenance (git commit, host, thread
  count — the wall-clock budget makes the machine an experimental parameter),
  and three plots (regret/dataset-size vs. duration, regret vs. response
  time, and a lengthscale/removal-budget diagnostic panel not in the paper).

## Synthetic (Ackley)

- Uses `ackley4d`: the standard Ackley test function with an appended
  temporal axis, per H.2's `z = (x_1, …, x_d, t)` construction — 3 spatial
  dimensions plus time, domain `[-32, 32]^4`.
- Fully analytic — no data download, no preprocessing. The objective and its
  oracle (`max_x f(x, t)`, via a dense spatial grid search per time slice) are
  computed and cached on first run.
- Ackley's spatial optimum sits at the origin for every `t` under this
  domain, so the "dynamic" difficulty is a ripple in achievable value rather
  than a moving target — an exploration/acquisition problem, not
  optimum-tracking (paper §5.2).
- Paper reference: average regret ≈ 2.24 (Table 2).

## Temperature

- A real-world benchmark from the Intel Berkeley Research Lab sensor
  dataset: ~54 deployed motes, reduced to 46–48 after dropping dead sensors
  and those with unreliable first-day delivery rates (a documented,
  reproducible coverage filter, since the paper's exact 46-sensor subset
  isn't published).
- Sensor readings are irregular in space and time, so they're fit into a
  continuous surface `f(x, y, t)` via RBF (thin-plate-spline) interpolation
  over the first calendar day of data; the task is finding the (space, time)
  location with the highest interpolated temperature as it drifts over the
  day.
- 2 spatial dimensions (normalized to `[0, 1]²`) + 1 temporal, per the
  paper's "3-dimensional benchmark" framing.
- Paper reference: average regret ≈ 0.68 (Table 2).
