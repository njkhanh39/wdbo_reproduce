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
- `alpha = 0.25`. §5.1's sensitivity analysis concludes: "the sweet spot is
  reached for α = ¼. This hyperparameter value is used to evaluate W-DBO in
  the next section" — so every Table 2 number is `α = 1/4`.
- 15 initial observations drawn uniformly from `S' × [0, 1/40]`, per H.1 —
  spread over the first fortieth of the horizon rather than stacked at
  `t = 0`. This matters: 15 observations at a single instant carry no
  information about the temporal lengthscale `lT`, which the removal budget
  `(1 + alpha) ** (dt / lT)` divides by. That window is charged against the
  600 s budget, so the optimization loop starts at `t = 1/40`. Initial
  observations are not queries the algorithm chose, so they do not appear in
  the regret log.
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

Results are written to a fresh timestamped directory
`data/temperature/results/<YYYYmmdd-HHMMSS>[-label]/`, so a run never
overwrites its predecessor; `--label paper` names it and `--results-dir`
overrides the path entirely. §5 explains every file it contains.

The optimization loop and everything downstream of it (the log, the
summaries, the run metadata, the plots) live in
[`../common.py`](../common.py), shared with the synthetic experiment so the
two cannot drift apart. This script only builds the objective and parses
flags.

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
- **Time-weighted average regret up to `t`**: the same idea, but each query
  is weighted by how long it actually stood before the next one replaced it —
  `sum(r_i · dt_i) / sum(dt_i)`, the discrete form of `(1/t)∫r dt`. The
  unweighted mean counts a query that stood for 3 s the same as one that
  stood for 0.3 s, whereas §5's protocol ("two iterations of a solution are
  separated by its response time") exists precisely so that being slow
  costs something. The paper never says which convention it uses, so both are
  reported; they agree when response times are stable and diverge exactly
  when cleaning stalls.
- **Response time** (`t_response`): real wall-clock seconds for the two tasks
  H.1 defines it as — (i) estimating the kernel and noise hyperparameters and
  (ii) optimizing the acquisition function. **Cleaning is not included**, and
  neither is querying the objective (H.1: "the objective function is
  immediately sampled"). Both still advance the wall clock — Algorithm 1
  reads the clock *after* the removal loop — they just aren't part of the
  metric. `t_clean` is logged separately; it is frequently comparable to
  `t_response` in size, so conflating them overstates response time badly.
- **Dataset size**: how many points W-DBO's internal model currently holds,
  after stale ones are cleaned out. Floored at 2 (the cleaning loop stops at
  `xx_tt.shape[0] > 2`).
- **`lT` and the removal budget**: the MLE temporal lengthscale and the
  budget `b` from Algorithm 1, both logged per query. The budget grows as
  `(1 + alpha) ** (Δt / lT)` — it *divides by* `lT` — so an `lT` the MLE
  drives towards zero makes the budget explode and W-DBO purge down to the
  floor of 2. When a run misbehaves, these are the two numbers to look at.
- **Dataset size**: how many points W-DBO's internal model currently holds,
  after stale ones are cleaned out. Floored at 2 (the cleaning loop stops at
  `xx_tt.shape[0] > 2`).
- **Replications / seeds**: each `--n-seeds` run is an independent rerun
  (see §4 for the seeding modes). "Averaged across runs" means: compute the
  metric once per replication, then average those replication-level numbers.

### 5.2 The files

A run leaves behind three CSVs, one JSON and three PNGs. **`queries.csv` is
the only irreplaceable one** — every other file is a view over it, and
[`../plot.py`](../plot.py) regenerates the figures from it without re-running
the experiment:

```bash
python experiments/plot.py data/temperature/results/20260914-120000-paper
```

#### `queries.csv` — the raw log

One row per real query, for **every** seed (a `seed` column distinguishes
them). No interpolation, no resampling, no fixed row count: a 600 s run makes
however many queries it makes, machine-dependent. Columns:

| column | meaning |
|---|---|
| `seed`, `iteration` | which replication, and the query's index within it |
| `sim_time` | the optimizer's normalized clock, `elapsed / duration` ∈ `[1/40, 1]` |
| `wall_time` | elapsed real seconds at the end of the step |
| `t_response` | H.1's response time: hyperparameter estimation + acquisition optimization |
| `t_clean` | seconds spent in `clean()` — advances the clock, but not part of `t_response` |
| `regret` | **instantaneous** regret of this query |
| `dataset_size` | model dataset size after cleaning |
| `n_removed` | how many observations this `clean()` call stripped |
| `lambda`, `lS`, `lT`, `noise` | the MLE hyperparameters the next iteration will use |
| `removal_budget` | Algorithm 1's budget `b` after this step |

The 15 initial observations are not in here: they are not queries the
algorithm chose, so they are excluded from the regret log (see §3).

#### `per_seed.csv` — one row per replication

`iterations`, both average-regret conventions, mean `t_response` and
`t_clean`, final/max/min dataset size, `median_lT`, and `total_removed`. This
is the file that shows "8 seeds fine, 2 stuck at 2" at a glance, and
`iterations` is the check on whether your machine is doing comparable work to
the paper's (an i9-9980HK, 8 cores / 16 threads).

#### `summary.csv` — the headline numbers

`metric, mean, sem, n_runs` for average regret, time-weighted average regret,
response time, clean time and iteration count. **Standard error, not
variance**: Table 2 underlines algorithms whose confidence intervals overlap
the best one's, so the SEM is what makes the comparison. The paper's W-DBO
figure for Temperature is **0.68**; the script prints it alongside your own.

#### `run.json` — provenance

Every CLI argument, the git commit and dirty flag, hostname, CPU model,
`torch.get_num_threads()`, and library versions. On a wall-clock-driven
benchmark **the machine is an experimental parameter**: a slower host fits
fewer iterations into the same 600 s and scores worse regret with no
algorithmic difference at all. Two runs are only comparable if this file
matches. Consequences worth acting on: never run the seeds concurrently (CPU
contention inflates response time and silently worsens regret), and keep the
thread count fixed across runs you intend to compare.

#### `regret_and_size_vs_duration.png` (2 panels)

Each seed's **running (cumulative) average** regret is computed first, *then*
resampled onto a shared 200-point `linspace(0, duration, 200)` wall-clock
grid (`np.interp`; values past a seed's last query are held flat), *then*
*then* summarised across seeds. That grid is a plotting internal — it is no
longer written to disk.

Both panels draw the **across-seed mean**, with every seed as a faint line
behind it. See §6.

- **Left — Regret**: average regret up to `t`. Mirrors Figure 20 (right),
  regret side.
- **Right — Dataset size**: log y-axis. See §6 for how to read the band.

#### `regret_vs_response_time.png` (1 panel)

Every query's `(t_response, instantaneous regret)` scattered across all
seeds, plus **one orange marker per seed** at that seed's own average and a
black X at the mean of those. Mirrors Figure 20 (left) — the paper overlays one
box per baseline algorithm; running W-DBO alone, the per-seed markers are what
show the spread.

#### `lengthscale_and_budget.png` (2 panels) — not in the paper

The diagnostic panel: `lT` (left) and the removal budget (right) against
duration, one faint line per seed plus the mean, both on log axes. This is
where a dataset collapse is explained rather than just observed.

## 6. Reading the plots, and the dataset collapse

Every panel plots the **across-seed mean** as the bold black curve — the same
statistic `summary.csv` reports — with each individual seed drawn faintly
behind it. The per-seed lines are there because on this benchmark the mean can
hide a split: a seed either keeps a healthy dataset (~90–150 observations) or
collapses to the cleaning floor of 2 and stays there. In the 10-seed run under
`results/20260914-140920-paper`, dataset size at `t = 600 s` was:

```
[2, 2, 2, 2, 2, 2, 49, 90, 123, 131]        mean = 40.6
```

The mean of 40.6 sits in the empty gap between the two groups. It is still what
gets plotted and quoted, but the faint lines make the split visible rather than
leaving it to be inferred.

### What the collapse is

Visible in `lengthscale_and_budget.png`, and worth knowing before you read any
result: the removal budget is `b *= (1 + alpha) ** (Δt / lT)` — it **divides
by** `lT`. The MLE in `model.py` is unconstrained, and it occasionally returns
a degenerate `lT` (values as low as `5e-5` appear in `queries.csv`, meaning the
objective decorrelates in ~30 ms of a 600 s run). One such fit multiplies the
budget by ~2000 in a single step. From the paper run, seed 3:

| it | wall | lT | budget | size | removed |
|---|---|---|---|---|---|
| 5 | 23.7 s | 1.00380 | 1.14 | 6 | 0 |
| **6** | 25.9 s | **0.00005** | 3.42 | **2** | **5** |
| 7 | 26.5 s | 0.00973 | 5.6e+08 | 2 | 1 |

It is an **absorbing state**. Once collapsed, the budget's median is `1.6e+16`
and `n_removed` is exactly 1.0 every step forever — one point added, one
deleted — and with 2 observations the GP can never re-estimate `lT` well enough
to escape. In that run 6 of 10 seeds collapsed; 3 died inside the first 50 s.

This is untouched upstream behaviour: `model.py` is the authors' own code and
has not been modified here. Fixing it would mean constraining the lengthscales
(a `GreaterThan` constraint or a lognormal prior, as botorch's own
`SingleTaskGP` ships by default) or raising the cleaning floor above 2. Neither
is done yet — for now the plots show the collapse honestly instead of averaging
it away.

Note the headline regret is largely unaffected: that run scored **0.603 ±
0.054** against the paper's 0.68, collapses and all.

### `--same-seed` runs

`saved/04_10_run_1_seed` and `results/20260915-092443-same_seed` reuse one seed
ten times. The replications still differ, because the loop is wall-clock driven
(§4), but they share initial points and the noise sequence, so they start
near-identical and diverge only as timing jitter compounds. They are a
diagnostic for that nondeterminism, not a paper-style result — and they inherit
whatever that one seed happens to be. Seed 0 is one of the collapsing seeds,
which is why the `same_seed` run scores **0.878 ± 0.096** against the
distinct-seed run's **0.603 ± 0.054**: same algorithm, ~45 % worse, purely from
the draw.

## 7. A note on `saved/`

The runs under `data/temperature/saved/` predate the output format described
in §5, and predate the corrections in §3 (`alpha`, the initial-observation
window, the response-time definition). They are kept as a record of earlier
attempts, but they are not comparable to new runs and `plot.py` cannot read
them — they have no `queries.csv` and no `run.json`.

## 8. What this reproduction does and doesn't match exactly

- **Matches**: the benchmark's definition (3D spatio-temporal, first day of
  data, activate-the-hottest-point task), the kernel choices, the 15 initial
  observations over `S' × [0, 1/40]`, `alpha = 1/4`, the 600 s wall-clock
  budget, and the noise model — all taken directly from Appendix H.1/H.2 of
  the paper.
- **Approximates**: the exact 46-sensor subset (we use a documented,
  reproducible 48-sensor filter instead — see §2); and the exact
  interpolation method used to build the ground-truth surface (the paper
  doesn't specify one; we use a thin-plate-spline RBF interpolator, a
  standard, deterministic choice for scattered spatio-temporal data).
- **Not included**: comparisons against the other baselines from the paper
  (GP-UCB, TV-GP-UCB, ABO, ET-GP-UCB, ...) — this reproduces W-DBO's own
  behavior on the benchmark, not the full comparative study.
