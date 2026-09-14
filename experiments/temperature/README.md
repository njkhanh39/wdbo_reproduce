# Reproducing the WDBO "Temperature" benchmark

This reproduces the real-world benchmark from Appendix H.2 of the WDBO paper
("This Too Shall Pass: Removing Stale Observations in Dynamic Bayesian
Optimization", Bardou, Thiran & Ranieri, 2024, https://arxiv.org/abs/2405.14540):

> Temperature. This benchmark comes from the temperature dataset collected
> from 46 sensors deployed at Intel Research Berkeley. [...] The goal of the
> DBO task is to activate the sensor with the highest temperature, which will
> vary with time. To make the benchmark more interesting, we interpolate the
> data in space-time [...] making it a 3-dimensional benchmark (2 spatial
> dimensions [...], 1 temporal dimension). For the numerical evaluation, we
> used the first day of data.

Concretely: sensor readings only exist at ~50 fixed points in space and at
irregular times, so the paper fits a continuous interpolated surface
`f(x, y, t)` over them and lets W-DBO query *any* point in space-time. The
DBO task is to keep finding the (space, time) location with the highest
interpolated temperature as the surface drifts over the day, using as few
"stale" past observations as possible.

This page is the single source of truth for the experiment: where the data
comes from, how the pipeline runs, what the `--seed` flag controls, and how
to read every number and plot it produces.

## 1. Where the data comes from

The raw data is the public [Intel Berkeley Research Lab dataset](https://db.csail.mit.edu/labdata/labdata.html):

- `data/temperature/data.txt` — ~2.3M raw readings (`date time epoch moteid temperature humidity light voltage`), collected from 2004-02-28 to 2004-04-05.
- `data/temperature/mote_locs.txt` — `moteid x y` table of the 54 sensor deployment locations.

Both files must be downloaded from the link above and placed in
`data/temperature/` (the `data/` folder is git-ignored — the raw dataset is
~150MB and not something we want in the repo). Every other artifact this
experiment produces or consumes also lives under `data/temperature/`; see
`paths.py` for the single `DATA_DIR` constant all scripts default to.

## 2. Why the paper says 46 sensors when the location table lists 54

This is worth being precise about, because the discrepancy is *not* a typo.

**Verified facts (see `preprocess.py`'s printed summary, or rerun the analysis
yourself):**

1. `data/temperature/mote_locs.txt` lists 54 sensor locations — this matches
   the official dataset page exactly.
2. Of those 54, **motes #5 and #28 never report a single reading in the
   entire ~5-week deployment** (not just the first day — checked against the
   full `data.txt`). These are dead-on-arrival sensors. That alone accounts
   for 54 → 52.
3. The remaining 52 sensors all report *some* data on the first day, but with
   very uneven delivery rates — this is a wireless sensor network, and motes
   farther from the base station lose a much larger fraction of their
   packets. Binning the first day into fixed-width time windows and looking
   at what fraction of windows each sensor actually reports in
   (`select_reliable_sensors` in `preprocess.py`) shows a clear, sharp gap
   that shows up at *every* bin width we tried (5/10/15/30 minutes):

   | mote | coverage | | mote | coverage |
   |---|---|---|---|---|
   | 20 | ~58% | | *(gap)* | |
   | 15 | ~66% | | 17 | ~94% |
   | 49 | ~72% | | 36 | ~95% |
   | 54 | ~73% | | ... (44 more, all ≥ 94%) | |

   Four sensors (15, 20, 49, 54) sit far below everyone else. Filtering at an
   85% coverage threshold — comfortably inside that gap — keeps **48**
   sensors.

**Why not exactly 46, then?** The "46 sensors" framing is not something the
WDBO authors derived themselves — the same benchmark (same wording, same
task) was used earlier by Bogunovic et al. (2016) and, in near-identical
language, by Brunzema et al. ("Event-Triggered Time-Varying Bayesian
Optimization", 2022/2025), whom the WDBO paper explicitly cites as its
source for this benchmark. Brunzema et al.'s own code repository
([github.com/brunzema/et-bo](https://github.com/brunzema/et-bo)) documents
the download/preprocessing steps in its README but **does not publish the
preprocessing script itself** (verified directly — the repo ships only a
README, license and an empty `examples/` folder). So the *exact* list of 46
sensors used upstream isn't recoverable from public sources.

What we did instead is the principled, transparent version of the same idea:
drop the two dead sensors, then drop the sensors whose first-day delivery
rate falls in the same "clearly worse" bucket that any reasonable
completeness filter would catch. That lands at 48 rather than 46 — close, and
built from a filter you can see and change (`--min-coverage` in
`preprocess.py`), rather than a hardcoded list of "the right" 46 IDs.

## 3. Pipeline

```
data/temperature/data.txt, data/temperature/mote_locs.txt
        │
        ▼
  preprocess.py   →  data/temperature/processed.npz
        │              (filtered, time-binned, [0,1]-normalized point cloud)
        ▼
  objective.py    →  interpolated ground-truth surface f(x, y, t)
        │              + oracle(t) = max_{x,y} f(x, y, t), cached to
        │                data/temperature/oracle.npz
        ▼
  run_experiment.py → runs WDBOOptimizer against the objective,
                       logs regret & dataset size, saves plots
```

### Step 1 — Preprocess the raw data

```bash
python experiments/temperature/preprocess.py
```

This reads `data/temperature/data.txt` + `data/temperature/mote_locs.txt`, restricts to the first
calendar day found in the dataset (2004-02-28), drops physically implausible
readings (a well-known artifact of this dataset: when a mote's battery
voltage sags, temperature/humidity readings become garbage — e.g. the very
first line of `data.txt` reports 122°C), averages readings into 10-minute
per-sensor bins, drops unreliable sensors (see §2), and normalizes
coordinates: spatial `(x, y)` to `[0, 1]²` over the kept sensors' bounding
box, and time-of-day to `[0, 1]`. It prints the sensor-count audit from §2
and writes `data/temperature/processed.npz`.

All thresholds are CLI flags, not hardcoded — run `--help` to see them
(date, bin width, coverage threshold, physical sanity ranges, output path).

### Step 2 — Run the experiment

```bash
python experiments/temperature/run_experiment.py
```

This builds the interpolated objective (`objective.py`, using
`scipy.interpolate.RBFInterpolator` — a one-off ~30-60s fit, then cheap to
query) and its oracle curve (a dense space-time grid search, cached to
`data/temperature/oracle.npz` so it's only computed once), then runs
`WDBOOptimizer` exactly as described in the paper's Appendix H.1:

- Matérn-5/2 spatial kernel, Matérn-3/2 temporal kernel.
- 15 initial random observations, `alpha = 0.25` (the paper's sensitivity
  analysis, §5.1, settles on `alpha = 1/3` for its headline Table 2 run;
  pass `--alpha 0.3333` to match that exactly).
- Each replication runs for a fixed real wall-clock budget (default 600s /
  10 minutes, matching the paper), during which the optimizer's internal
  clock sweeps linearly over the day (`current_time = elapsed / duration`).
  This mirrors `src/test.py`'s own real-time-driven loop, just rescaled to
  cover a full simulated day instead of a few real minutes.
- Observations are corrupted with Gaussian noise at 5% of the signal
  variance, per Appendix H.1.

Pass `--n-seeds 10` to average across independent replications like the
paper does (each replication reruns the whole optimization from scratch with
a different seed, and results are interpolated onto a common duration grid
before averaging); the default is 1 for a quick single run. The paper's own
figures use 10 replications, so `--n-seeds 10` is what an actual "official"
reproduction run should use — it just takes ~10x longer.

Two seeding modes:

- **default** (`--seed S`): replication *i* uses seed `S + i` — 10 genuinely
  independent runs, as in the paper.
- **`--same-seed --seed S`**: every replication uses seed `S`. The runs are
  then *not* identical (see §4), but they are highly correlated; this is a
  diagnostic mode for isolating wall-clock nondeterminism, not a paper-style
  run. The saved run `04_10_run_1_seed` was produced this way.

Results are written to `data/temperature/results/`: 2 CSVs (`regret.csv`,
`summary.csv`) and 2 plot files holding 3 panels total (regret + dataset
size vs. duration, and regret vs. response time) — reproducing both halves
of Figure 20. §5 explains exactly what each file and metric means.

## 4. What the `--seed` flag controls

`run_once()` seeds three RNGs from the same integer:

| RNG | Feeds |
|---|---|
| `np.random.default_rng(seed)` → `rng` | **Observation noise only** — `y = true_value + rng.normal(0, noise_std)` |
| `np.random.seed(seed)` (global NumPy) | The **15 initial random query points** — `np.random.uniform(...)` in `optimizer.next_query` |
| `torch.manual_seed(seed)` (global Torch) | The **random restarts of the acquisition optimizer** — `optimize_acqf`'s `raw_samples=512` / `num_restarts=20`. GP fitting itself is Adam from deterministic inits, so it's reproducible given identical data. |

**What the seed does *not* pin down — and why `--same-seed` runs still
diverge:** the loop is driven by real elapsed wall-clock time
(`current_time = (now - start) / duration_seconds`). Machine load, GC
pauses, and how long each GP fit / clean happens to take all change *at
which simulated times* queries land and *how many* iterations a replication
completes. Two `--same-seed` replications share the initial points and the
noise draw sequence, but `optimize_acqf` then gets called with a
different `current_time` bound, picks a different candidate, and the runs
drift apart — a drift that compounds over the run.

Also outside the seed's control (fully deterministic): the interpolated
objective and its cached oracle curve.

## 5. Reading the results

### 5.1 The metrics — don't mix these up

- **Instantaneous regret**: the regret of *one single query*. At the moment
  W-DBO picks a point `x` at time `t`, instantaneous regret = `best(t) -
  f(x, t)` (best possible value at that instant, minus what we actually
  got). One value per query.
- **Average (running) regret up to `t`**: the *cumulative mean* of every
  instantaneous regret from all queries made so far, up to time `t`. This is
  the `R_t / t` convention: it keeps averaging in more queries as `t` grows,
  so it smooths out and typically trends toward a stable value. **This is
  the paper's convention and the one to compare against Figure 20.**
- **Response time**: real wall-clock seconds one iteration took (pick a
  point + update the model + clean stale data). Not regret — a speed metric.
- **Dataset size**: how many points W-DBO's internal model currently holds,
  after stale ones are cleaned out. Floored at 2 (the cleaning loop stops at
  `xx_tt.shape[0] > 2`).
- **Replications / seeds**: each `--n-seeds` run is an independent rerun
  (see §4 for the seeding modes). "Averaged across runs" means: compute the
  metric once per replication, then average those replication-level numbers.

### 5.2 The files

#### `regret.csv`

A time series, but **not** a raw query log when `--n-seeds > 1`.

- **`--n-seeds 1`**: the file *is* the raw log — one row per real iteration
  (so the row count is however many queries that run actually made, *not* a
  round number), with 5 columns: `sim_time, wall_time, response_time,
  regret, dataset_size`. `regret` here is instantaneous.
- **`--n-seeds > 1`** (how `03_modified_10_iters` and `04_10_run_1_seed`
  were made): the file has exactly **200 rows** and 3 columns: `wall_time,
  regret, dataset_size`. Those 200 rows are a *resampling*, not queries:

  1. Each replication logs `(wall_time, instantaneous_regret, dataset_size)`
     per query; replications differ in iteration count and in the wall
     times they hit.
  2. Build a fixed grid `np.linspace(0, duration_seconds, 200)` — 200
     equally-spaced wall-clock times.
  3. For **each replication**, piecewise-linearly interpolate its regret and
     dataset size onto those 200 times (`np.interp`; values past a run's
     last logged time are held flat).
  4. **Average across the replications**: row *i* is the mean over the seeds
     of the interpolated value at grid time *i*.

  So "200" is a hardcoded plotting resolution, and the `wall_time` column is
  literally `linspace(0, 600, 200)` — `0, 3.015…, 6.030…, …, 600`. The
  algorithm does *not* query exactly 200 times; a real 600s run does a few
  hundred iterations, machine-dependent.

  Note this `regret` column is still **instantaneous** regret (interpolated,
  then seed-averaged) — *not* the running average. The running-average curve
  is computed separately for the plot (see below).

#### `summary.csv`

Two headline numbers, each as `mean, variance` across replications
(`ddof = 1` when there is more than one run):

1. **Average regret up to `t = duration_seconds`** — for each replication,
   the mean of *all* its instantaneous regrets over the whole run; then mean
   and variance of those per-replication numbers. This is the value the left
   plot panel reaches at its rightmost point.
2. **Average response time** — the mean per-query response time within each
   replication, then mean and variance across replications.

This is the file to quote when reporting "one number" for the whole run.

#### `regret_and_size_vs_duration.png` (2 panels)

Both panels come from `compute_duration_stats`, which — unlike `regret.csv`
— computes each replication's **running (cumulative) average** regret first,
*then* interpolates onto the 200-point grid, *then* takes the mean and std
across replications.

- **Left — Regret**: running average regret up to `t` on the y-axis,
  duration (seconds) on the x-axis. Shaded band = ±1 std. dev. across
  replications. Mirrors Figure 20 (right), regret side.
- **Right — Dataset size**: dataset size on a **log** y-axis vs. duration,
  same shaded-band convention. See §6 for how to read the band on this
  panel — it can look alarming and usually isn't.

#### `regret_vs_response_time.png` (1 panel)

Scatter of every individual query's `(response time, instantaneous regret)`
across all replications (light dots), plus one bold marker at the mean
response time / mean regret with error bars (±1 std across replications).
Mirrors Figure 20 (left) — the paper overlays one such box per baseline
algorithm; since this script only runs W-DBO, there's just one marker here.

## 6. Interpreting the uncertainty bands

**The dataset-size band on the log plot can plunge to the bottom of the
axis. That is a plotting artifact, not a collapse of the dataset.** The
lower edge of the band is `np.clip(size_mean - size_std, 1e-6, None)` drawn
on a log scale. Whenever `size_std ≥ size_mean` at a grid point,
`mean - std` is ≤ 0, gets clipped to `1e-6`, and the shaded region visually
falls to the floor of the plot. The real dataset size never goes near zero —
W-DBO's cleaning loop is hard-floored at 2.

**Why the band is thin early and explodes late in a `--same-seed` run
(`04_10_run_1_seed`).** All 10 replications start essentially identical
(same initial points, same noise sequence) → std ≈ 0, a razor-thin band.
They diverge *only* through wall-clock nondeterminism (§4), and that
divergence compounds, so by the last third the runs have drifted far apart —
visible in *both* the regret band widening and the size band. On top of
that, W-DBO's removal budget grows multiplicatively with elapsed time
(`budget *= (1 + alpha) ** (Δt / lT)`), so once it is large a single
`clean()` call can strip many points at once. If even 1–2 of the 10
replications undergo an aggressive purge near the end while the other 8 sit
at 50–70 points, the cross-run std becomes comparable to the mean → the clip
above fires → the band drops to the axis floor. The mean stays "consistent"
(~50) because it is dominated by the replications that did *not* purge.
`np.interp` holding each run's final value flat to `t = 600` smears this
across the last few grid points rather than a single one.

**A `--n-seeds 10` run with distinct seeds (`03_modified_10_iters`)** does
not show this "thin then exploding" pattern: the replications differ from
step 1, so the band is broad and roughly uniform throughout, and
`mean - std` for dataset size stays positive — no clip-to-`1e-6` crash.

## 7. What this reproduction does and doesn't match exactly

- **Matches**: the benchmark's definition (3D spatio-temporal, first day of
  data, activate-the-hottest-point task), the kernel choices, initial
  observation count, wall-clock experiment budget, and noise model — all
  taken directly from Appendix H.1/H.2 of the paper.
- **Approximates**: the exact 46-sensor subset (we use a documented,
  reproducible 48-sensor filter instead — see §2); the exact interpolation
  method used to build the ground-truth surface (the paper doesn't specify
  one; we use a thin-plate-spline RBF interpolator, a standard,
  deterministic choice for scattered spatio-temporal data); and `alpha`
  (script default `0.25` vs. the paper's `1/3` headline value — see §3).
- **Not included**: comparisons against the other baselines from the paper
  (GP-UCB, TV-GP-UCB, ABO, ET-GP-UCB, ...) — this reproduces W-DBO's own
  behavior on the benchmark, not the full comparative study.
