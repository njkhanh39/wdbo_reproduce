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
- 15 initial random observations, `alpha = 0.25`.
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

Results are written to `data/temperature/results/`: 2 CSVs (`regret.csv`,
`summary.csv`) and 2 plot files holding 3 panels total (regret + dataset
size vs. duration, and regret vs. response time) — reproducing both halves
of Figure 20. **See [help.md](help.md) for exactly what each file/metric
means** (in particular, instantaneous vs. average/running regret — easy to
mix up).

## 4. What this reproduction does and doesn't match exactly

- **Matches**: the benchmark's definition (3D spatio-temporal, first day of
  data, activate-the-hottest-point task), the kernel choices, initial
  observation count, `alpha`, wall-clock experiment budget, and noise model
  — all taken directly from Appendix H.1/H.2 of the paper.
- **Approximates**: the exact 46-sensor subset (we use a documented,
  reproducible 48-sensor filter instead — see §2) and the exact
  interpolation method used to build the ground-truth surface (the paper
  doesn't specify one; we use a thin-plate-spline RBF interpolator, a
  standard, deterministic choice for scattered spatio-temporal data).
- **Not included**: comparisons against the other baselines from the paper
  (GP-UCB, TV-GP-UCB, ABO, ET-GP-UCB, ...) — this reproduces W-DBO's own
  behavior on the benchmark, not the full comparative study.
