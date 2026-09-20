# Experiment Design

This directory reproduces **W-DBO** — Wasserstein-based Dynamic Bayesian
Optimization, from Bardou, Thiran & Ranieri, *"This Too Shall Pass: Removing
Stale Observations in Dynamic Bayesian Optimization"* (2024) — and extends it
with alternative removal rules so the paper's central claim can actually be
tested: **that removing stale observations helps, and that the Wasserstein
distance is the right criterion for deciding which ones to remove.**

This page is the design document: what is being measured, how the clocks work,
what each knob does, and what a row of the log means. For results, known
issues and benchmark-specific detail, see
[`synthetic/README.md`](synthetic/README.md) and
[`temperature/README.md`](temperature/README.md). For the MI removal rule, see
[`MI_CRITERION.md`](MI_CRITERION.md). For what recently changed in the harness
and why — the environment clock, the re-scorable log, and what old results are
still good for — see [`CHANGES.md`](CHANGES.md).

---

## 1. What the experiment is

Dynamic Bayesian Optimization maximizes an objective that **moves**:
`f(x, t)`, where `x` is a configuration you choose and `t` is time you do not
control. A GP is fit over space *and* time; the acquisition function proposes
the next `x`; the observation is added to the dataset; the loop repeats.

The problem W-DBO addresses is that this dataset grows without bound, and GP
inference is `O(n^3)`. Old observations describe a world that no longer
exists, so they cost cubic time to carry while contributing less and less.
W-DBO's answer is to **delete** observations whose removal would barely change
the model's predictions about the future, measured by a Wasserstein distance
between the posteriors with and without them.

Which makes the experiment a race against a clock rather than a fixed number
of iterations. Every arm gets the same wall-clock budget and the same moving
objective; an arm that keeps more data makes better-informed choices but fewer
of them. **That trade-off is the measurement.** Comparing arms at a fixed
iteration count would hide exactly the cost being studied — see the warning in
`run_once`'s docstring.

### The arms

All three are the same optimizer, the same kernels, the same acquisition
function and the same clock. Only the removal rule differs.

| `--criterion` | Rule | Notes |
|---|---|---|
| `wasserstein` | W-DBO's own criterion | The paper's method. `--alpha 0.25` is its Table 2 setting. |
| `mi` | Mutual information with the future maximum | Our alternative. Unnormalized (nats), so `--mi-alpha` is **not** comparable to `--alpha`; see [`MI_CRITERION.md`](MI_CRITERION.md). |
| `none` | Never call `clean()` | The ablation the paper lacks. The dataset grows monotonically. |

The `none` arm is what makes the paper's claim falsifiable. The paper compares
W-DBO against other *algorithms*; it never compares W-DBO against **itself
with removal switched off**, so "removal helps" is assumed rather than shown.
`--criterion none` (or its alias `--no-removal`) is that missing control.

---

## 2. Two clocks

This is the part most likely to trip you up, and the part most recently
changed. The loop keeps **two independent clocks**.

### The wall clock — how much compute

Real seconds, read with `time.perf_counter()`. It starts at `0` when the
optimization loop proper begins and stops at `--duration-seconds` (`H`,
default 600 s, the paper's budget). This is the **compute budget**: how much
thinking the algorithm is allowed to do.

### The environment clock — how fast the world moves

`t_env`, in the objective's own time units. It advances as

```
t_env = env_start + env_speed * elapsed_seconds
```

where `--env-speed` is in **environment units per real second**. This is the
**difficulty**: how fast the target drifts away from you.

### Why they are separate

They used to be the same thing. The old loop computed
`current_time = elapsed / duration_seconds`, so the environment clock was
always `[0, 1]` regardless of the budget. That meant:

| Budget | Environment traversed | Implied speed |
|---|---|---|
| 600 s | the whole temporal domain | `span / 600` |
| 200 s | the whole temporal domain | `span / 200` — **3× faster** |

So a 200 s run was not "the same experiment with less compute". It was a
**different, three times harder problem**, and any comparison between the two
confounded budget with difficulty. You could not answer "does more compute
help?" because more compute also made the problem easier.

With the two clocks separated:

- change `--duration-seconds` → same world, watched for longer or shorter;
- change `--env-speed` → same budget, a world that moves faster or slower.

**Move one at a time.**

### Defaults reproduce the paper

`common.default_env_speed` returns the speed at which a 600 s run covers the
benchmark's whole temporal domain exactly — the paper's setting. So a default
run is unchanged in spirit, and every other duration observes *that same
environment* for a different length of time. Concretely:

| Benchmark | Temporal domain | Default `--env-speed` |
|---|---|---|
| `ackley4d` | `[−32, 32]` (H.2's box) | `64 / 615 ≈ 0.10407` units/s |
| `temperature` | `[0, 1]` (one calendar day) | `1 / 615 ≈ 0.0016260` units/s |

The `615 = 600 × (1 + 1/40)` accounts for the initial design, below.

### The initial design and the warm-up

Appendix H.1 seeds the optimizer with 15 observations drawn uniformly over
`S' × [0, 1/40]` of the temporal domain — spread over a window, not stacked at
one instant, because observations gathered at a single time carry no
information about the temporal lengthscale `lT`, which the removal budget
`(1 + alpha)^(dt / lT)` divides by.

Two things follow, and both changed recently:

1. **The warm-up's environment window is sized against the 600 s reference
   run, not against this run's `H`.** It is `(1/40) × env_speed × 600`
   environment units, so it depends only on the speed. That keeps `env_start`
   identical across durations, which makes **a 200 s run an exact prefix of a
   600 s run** at the same speed. Scaling it with `H` would have reintroduced
   the coupling in miniature.

2. **Its real cost is measured, not assumed.** The old loop back-dated the
   wall clock by `H/40` seconds, i.e. *pretended* the warm-up took exactly
   15 s of a 600 s run whatever it really took. Now the initial design is
   treated as a warm start handed to the algorithm — a dataset you already
   had — the wall clock starts at zero once it is in hand, and the seconds it
   cost are reported separately as `warmup_seconds` in `per_seed.csv`. They
   are charged to nobody.

### The deadline

`H` is a hard stop checked **before** each query is committed to, not after.
A query started with no budget left would be scored against an environment the
run was never supposed to reach. An iteration that starts inside the budget is
allowed to finish (killing a half-conditioned GP would corrupt the run), and
that fact is recorded as `overran` in the run info, so a scoring pass knows the
last row runs past the end and must be clipped at `H`.

---

## 3. The oracle

Regret needs `f*(t) = max_x f(x, t)`, the best value achievable at time `t`.
Both benchmarks precompute it by dense grid search over space at many times,
cache it to `.npz`, and read it back with `np.interp` in the hot loop.

**The table is indexed by absolute environment time.** This matters more than
it sounds. The old table was indexed by the *run's* normalized `[0, 1]` clock,
with the map onto the function's axis folded in at build time. That made the
oracle a property of the run rather than of the function, with two
consequences:

- every environment span needed its own cache file; and
- because the sample **count** was fixed per file, the table's resolution *per
  unit of environment time* silently changed whenever the span did.

The second one is a real bug, not an inconvenience. On an oscillatory
benchmark, an under-sampled `f*` **misses peaks**, so it is biased *low*, so
regret comes out **better than it really is** — and it fails silently.

So `--oracle-density` is in **samples per unit of environment time**.
Resolution is now a property of the function, and one table serves every
`(env_speed, duration)` pair whose interval falls inside it.

| Benchmark | Default density | Rationale |
|---|---|---|
| `ackley4d` | 64 /unit | Ackley's `cos(2πz)` term completes one period per unit of time, so this is ~64 samples per period. |
| `temperature` | 200 /unit | The interpolated sensor surface is far smoother in time. |

Outside the cached span `np.interp` **clamps** rather than raising, which
would report a plausible-looking but wrong regret. So both objectives expose
`assert_covers(env_start, env_end)`, called once per run before any compute is
spent:

- **synthetic** — widen the table with `--env-span LO HI` (rebuilds the cache);
- **temperature** — there is no fix. The sensor data stops at `t = 1`, and
  `RBFInterpolator` will extrapolate past it and return confident nonsense. A
  run longer than 600 s must lower `--env-speed` proportionally.

Before trusting a new `--env-speed`, double the density and check that `f*`
does not move.

---

## 4. Metrics

### Instantaneous regret

`regret = f*(t_env) − f(x, t_env)`, computed against the **noise-free** value
of the point actually queried. Using the noisy observation would credit an
algorithm for lucky noise draws: if the true value is 10 and the reading comes
back 12, the regret is measured against 10.

### Average regret over queries — `avg_regret`

The plain mean over queries. This is what the paper's Table 2 most likely
reports, and what to quote when comparing against it. It says how good the
*chosen points* were, but it ignores time entirely: a slow arm that made 50
excellent choices and a fast one that made 500 equally good choices score the
same, even though the slow arm left the system badly configured for most of
the run.

### Average regret over time — the one that matters for a live system

```
R_time = (1/H) ∫₀^H [ f*(t) − f(x_held(t), t) ] dt
```

where `x_held(t)` is the configuration **actually in force** at time `t`,
including while the algorithm is still thinking or waiting for a measurement.
This is the right headline metric for a system that must run continuously: it
charges an arm for every second it sat on a stale configuration.

**The scorer is not written yet, but the log now supports it.** Computing it
means re-evaluating `f(x_held(t), t)` on a fine time grid after the run, which
needs to know which `x` was in force when. `queries.csv` records exactly that:
every `x`, and the wall-clock instant it took effect.

The convention: **`x_i` takes effect at `t_apply_i`** — the moment the
acquisition optimization finishes and `f` is sampled — and holds until
`t_apply_{i+1}`. While the optimizer is choosing `x_i`, the *previous*
configuration is still running. The opening stretch `[0, t_apply_0)` is
covered by the `INITIAL_ROW`. So:

```python
runs, meta = load_run(results_dir)
objective = load_objective(meta)          # rebuilds f and f* exactly as the run had them
sch, H = meta["env_schedule"], meta["args"]["duration_seconds"]

for run in runs:                          # includes the INITIAL_ROW at index 0
    apply_t = np.array([r["t_apply"] for r in run])
    points = np.stack([query_point(r) for r in run])

    grid = np.linspace(0.0, H, 4000)      # clipped at H, ignoring any overrun
    held = np.clip(np.searchsorted(apply_t, grid, side="right") - 1, 0, len(run) - 1)
    env = sch["env_start"] + sch["env_speed"] * grid

    inst = np.array([objective.oracle(t) - objective.evaluate(points[i], t)
                     for t, i in zip(env, held)])
    r_time = np.trapezoid(inst, grid) / H
```

`load_objective(metadata)` rebuilds the objective at the same oracle density
and grid resolution the run used, so recomputed regret matches the log's own
`regret` column exactly. Use it rather than constructing the objective by
hand: both benchmarks ship a module literally named `objective`, so a scorer
that handles both and does a plain `import objective` gets whichever directory
came first on `sys.path` — scoring one benchmark against the other's function.
`load_objective` loads each under a unique name to make that impossible.

> ### `time_weighted_avg_regret` is deprecated — do not quote it
>
> The column still in `summary.csv` computes `Σ rᵢ·Δtᵢ / Σ Δtᵢ`. It is wrong
> in three separate ways, kept only so the new code stays comparable to the
> numbers already recorded in `logs/` while the real scorer is written.
>
> 1. **It multiplies a snapshot by a duration.** `rᵢ = f*(tᵢ) − f(xᵢ, tᵢ)` is
>    measured at one instant. Multiplying by `Δtᵢ` assumes regret holds still
>    while `xᵢ` is in force. In a dynamic problem it does not: *both* `f*(t)`
>    and `f(xᵢ, t)` drift, and they drift **apart** — a fixed point gets left
>    behind as the surface slides under it. That growth is precisely what
>    makes a stale configuration bad, and this metric is blind to it.
> 2. **It is misaligned by one step.** `wall_time` is stamped at the *end* of
>    the step, so `Δtᵢ` spans the interval ending after `xᵢ` has already been
>    cleaned up after — most of which `xᵢ` did not yet exist for. The interval
>    `xᵢ` actually stood is `[t_queryᵢ, t_queryᵢ₊₁)`. The practical damage:
>    a slow `clean()` after query `i` inflates `Δtᵢ₊₁`, so it penalizes
>    `xᵢ₊₁` instead of `xᵢ`, which is the point that actually had to wait.
> 3. **It is normalized by `Σ Δt`, not by `H`.** The stretch after the last
>    query is never counted, so arms making different numbers of queries are
>    normalized over slightly different horizons.

### Speed

`t_acq_fit` is Appendix H.1's response time: acquisition optimization (ii)
plus GP conditioning and hyperparameter re-estimation (i). The parts are
logged separately too, so a slow arm can be blamed on the right stage:

| Column | What it times |
|---|---|
| `t_acq` | optimizing the acquisition function — H.1's (ii) |
| `t_fit` | `tell()`: conditioning the GP and re-estimating hyperparameters — H.1's (i) |
| `t_acq_fit` | `t_acq + t_fit`, i.e. H.1's response time |
| `t_eval` | querying `f` — the harness's cost, not the algorithm's, but it still advances the wall clock |
| `t_clean` | the removal loop — advances the clock, but explicitly *not* part of response time |

Keeping `t_clean` out of the response time follows H.1, which defines it as
(i) + (ii) only. Keeping it *logged* is how you find out that an arm's
cleaning rule is eating its own budget.

---

## 5. Shared setup

- **Optimization loop, logging, and plotting** all live in
  [`common.py`](common.py). The two benchmarks differ *only* in how they build
  their objective, so their arms cannot drift apart.
- **An objective** is anything exposing `spatial_domain`, `noise_std`,
  `evaluate(x, t_env)`, `oracle(t_env)` and `assert_covers(start, end)`.
- **Kernels**: Matérn-5/2 over space, Matérn-3/2 over time.
- **Removal floor**: `--min-dataset-size`, default **15** (the initial-design
  size). Note the original W-DBO lets the dataset fall to **2**; pass
  `--min-dataset-size 2` for that behaviour. This default is ours, and it
  matters — see the temperature README's section on dataset collapse.
- **Removal hyperparameter**: `--alpha 0.25`, the paper's reported sweet spot
  (§5.1) and the value behind every Table 2 number.
- **Noise**: Gaussian at 5% of the objective's signal variance (H.1).
  Estimated over the benchmark's *whole* temporal domain, not the sub-interval
  a given run visits, so the noise level is identical across durations and
  speeds.
- **Replication**: `--n-seeds 10`, each with its own seed, noise draw, initial
  points and acquisition restarts. Reported with standard error, since Table 2
  underlines algorithms whose confidence intervals overlap the best.
- **The machine is an experimental parameter.** On a wall-clock benchmark a
  slower host completes fewer iterations in the same 600 s and scores worse
  regret with no algorithmic difference at all. `run.json` therefore records
  the host, CPU, thread count, git commit and library versions. Only compare
  runs from the same machine.

---

## 6. Outputs

Each run writes a timestamped directory under
`data/<benchmark>/results/<timestamp>[-label]/`:

| File | Contents |
|---|---|
| `queries.csv` | **The raw log, one row per query.** The only irreplaceable artifact. |
| `per_seed.csv` | One row per replication: the summary stats plus `warmup_seconds`, `elapsed_seconds`, `overran` and the environment interval covered. |
| `summary.csv` | Mean ± standard error across seeds, for the quotable metrics. |
| `run.json` | Provenance: all arguments, the resolved environment schedule, git commit, host, thread count, library versions. |
| `*.png` | Regret and dataset size vs. duration; regret vs. response time; an `lT`/removal-budget diagnostic panel (not in the paper). |

Everything except `queries.csv` is a *view* over it. A replication costs ten
minutes, so change a plot and re-render with
[`plot.py`](plot.py) — never re-run the experiment.

### `queries.csv` columns

| Column | Meaning |
|---|---|
| `seed`, `iteration` | replication id; `iteration` runs `−1, 0, 1, …` per replication, which is how `load_run` splits them |
| `env_time` | **absolute environment time** the query was issued at |
| `wall_time` | real seconds since the loop started, stamped at the *end* of the step (`== t_step_end`) |
| `t_iter_start` | when the iteration began, i.e. when `env_time` was read |
| `t_apply` | **when `x` took effect** — acquisition done, `f` being sampled. The timestamp `x_held(t)` is built from |
| `t_result` | when the reading was in hand |
| `t_update_done` | when `tell()` returned |
| `t_acq`, `t_eval`, `t_fit`, `t_acq_fit`, `t_clean` | durations of each stage — the timing split above |
| `x_0 … x_{d−1}` | the configuration queried, in the benchmark's own units |
| `y` | the noisy reading the algorithm saw |
| `true_value` | `f(x, env_time)`, noise-free — what regret is measured against |
| `regret` | `f*(env_time) − true_value` |
| `dataset_size`, `n_removed` | after this iteration's `clean()` |
| `lambda`, `lS`, `lT`, `noise` | MLE hyperparameters as the *next* iteration will see them |
| `removal_budget` | `(1 + alpha)^(dt / lT)`; `clean()` removes while this exceeds 1 |
| `min_criterion`, `criterion_lT` | the cheapest observation's score, and the `lT` it was scored under (pre-cleaning, unlike the `lT` column) |
| `budget_spent` | budget consumed by this iteration's removals |

All timestamps are elapsed seconds from the loop's zero. Map them onto
environment time with `env_start + env_speed × t`, both of which are in
`run.json` and `per_seed.csv`.

Regret is computed in the loop as a cheap diagnostic, and against the
**noise-free** value: crediting an algorithm for a lucky noise draw would be
wrong. But the loop's convention is frozen at run time, whereas the log can be
re-scored under any convention — which is the point of logging `x`.

### The `iteration = −1` row

Each replication opens with one `INITIAL_ROW`: the last observation of the
initial design. It is the configuration in force from wall-clock zero until
the first real query applies, so a scorer cannot integrate the opening stretch
without it.

It is **excluded from every metric** (`common.queries_only`) — it is a warm
start the algorithm was handed, not a point it chose — and its stage timings
are `NaN`, because it predates the wall clock. If you write your own analysis,
filter it out before averaging anything.

### Reading older results

| Written before | Effect |
|---|---|
| the environment clock | `sim_time` / `t_response` are renamed to `env_time` / `t_acq_fit` on load, so `plot.py` works. Environment speed was `span / duration_seconds`, so the numbers are **not comparable** to new ones. |
| the `x` log | No `x`, `y`, `true_value` or stage timestamps, so the run **cannot be re-scored** — the whole point of the change. Split timings come back `NaN`. |

Both fall out the same way in practice: **old `queries.csv` files still plot,
but cannot be compared to or re-scored alongside new ones.** Re-run anything
whose numbers you intend to quote. Do not relabel an old directory and carry
it forward.

---

## 7. The benchmarks

### Synthetic (Ackley)

`ackley4d`: the standard Ackley function with its last axis reinterpreted as
time, per H.2's `z = (x₁, …, x_d, t)` construction — 3 spatial dimensions plus
time, on `[−32, 32]^4`. Fully analytic; no download, no preprocessing.

Ackley's spatial optimum sits at the origin for **every** `t` under this
domain, so the "dynamic" difficulty is a ripple in the achievable *value*
rather than a moving target — an exploration/acquisition problem, not
optimum-tracking (paper §5.2). Keep that in mind when reading results: this
benchmark cannot test whether an arm tracks a drifting optimum, which is a gap
section 4 of the benchmark note asks us to fill with new test functions.

Paper reference: average regret ≈ **2.24** (Table 2).

### Temperature

The Intel Berkeley Research Lab sensor dataset: ~54 deployed motes, reduced to
46–48 after dropping dead sensors and those with unreliable first-day delivery
rates (a documented, reproducible coverage filter — the paper's exact
46-sensor subset is not published).

Readings are irregular in space and time, so they are fit into a continuous
surface `f(x, y, t)` by RBF (thin-plate-spline) interpolation over the first
calendar day; the task is to track the hottest location as it drifts. 2 spatial
dimensions (normalized to `[0, 1]²`) plus time, per the paper's
"3-dimensional benchmark" framing.

Paper reference: average regret ≈ **0.68** (Table 2).

---

## 8. Running it

```bash
# The paper's setting: 600 s, 10 seeds, W-DBO's own criterion.
python experiments/synthetic/run_experiment.py --n-seeds 10 --label paper

# The missing control: same everything, removal switched off.
python experiments/synthetic/run_experiment.py --n-seeds 10 --criterion none --label ablation

# Same environment, a third of the compute. Now a meaningful comparison.
python experiments/synthetic/run_experiment.py --duration-seconds 200 --label short

# Same compute, an environment that moves twice as fast.
python experiments/synthetic/run_experiment.py --env-speed 0.208 --label fast

# Temperature: preprocess once, then run.
python experiments/temperature/preprocess.py
python experiments/temperature/run_experiment.py --n-seeds 10 --label paper

# Re-render figures from an existing log.
python experiments/plot.py data/synthetic/ackley4d/results/20260920-120000-paper
```

`python experiments/<benchmark>/run_experiment.py --help` lists every flag
with its default.

---

## 9. Known gaps

Tracked against the benchmark note's priorities:

- **No post-hoc scoring pass.** The log now carries everything needed
  (`x`, `y`, `true_value`, stage timestamps, the `INITIAL_ROW`), but nothing
  reads it back to compute `(1/H) ∫ [f*(t) − f(x_held(t), t)] dt` yet. Until
  that lands, the headline number is still `avg_regret`, which ignores time.
- **`time_weighted_avg_regret` is wrong** and still printed. It goes once the
  real scorer exists.
- **Ackley does not move its optimum**, so neither benchmark currently tests
  optimum-tracking, dwelling in a persistently suboptimal region, or abrupt
  regime change.
- **Correctness tests are not written.** The three the note calls for:
  `f(x, t) = −(x − t)²` with `x` held at 0 should give time-averaged regret
  → 1/3; a static function with an optimal configuration should give 0 regret
  however long you stall; and the `none` arm must log and score correctly
  despite never calling `clean()`.
