# Changes: the environment clock and the re-scorable log

**Current status:** Priority 3 is implemented in `scoring.py` and the shared
reporting path. The historical notes below describe the state after priorities
1 and 2; references to the scorer as "still missing" refer to that revision.

What was changed in the experiment harness to address **priorities 1 and 2** of
§5 of the benchmark note (`Benchmark_DBO.pdf`, 19/09/2026), why, and what it
unlocks. For the experiment design itself see [`README.md`](README.md).

Priority 3 (the post-hoc scoring pass) and the Dual Gate arm are not covered
here — they are a separate piece of work.

---

## Priority 1 — The clock

### Environment speed is now separate from run duration

The loop advanced its clock as `current_time = elapsed / duration_seconds`, so
the environment always swept its whole temporal domain regardless of the
budget. Shortening a run from 600 s to 200 s therefore did not just cut the
compute budget — it made the environment drift **three times faster**. Budget
and difficulty were the same knob, so no comparison could separate them.

Now:

```
t_env = env_start + env_speed * elapsed_seconds
```

`--env-speed` is in environment units per real second and is independent of
`--duration-seconds`. The default reproduces the paper's setting (a 600 s run
covers the benchmark's whole temporal domain), and the initial-design window is
sized against that same 600 s reference rather than against the current run —
so at a fixed speed, **a 200 s run is an exact prefix of a 600 s run**.

### Time accounting

| Before | After |
|---|---|
| Wall clock back-dated by `H/40`, i.e. the warm-up was *assumed* to cost exactly a fortieth of the run | Warm-up measured and reported separately as `warmup_seconds`. Measured values: 3.02 s for the first seed (Torch start-up) vs 0.27 s for later ones — the old assumption was wrong in both directions |
| `time.time()` | `time.perf_counter()` |
| No deadline check; an iteration starting past `H` was still scored | Deadline checked **before** each query is committed to; an iteration that starts inside the budget may finish outside it, and that is recorded as `overran` |
| `t_response` only | `t_acq`, `t_fit`, `t_eval`, `t_clean` logged separately; `t_acq_fit` remains H.1's response time |

### The oracle is indexed by absolute environment time

`--oracle-density` is now **samples per unit of environment time**, not samples
per table. Previously the sample count was fixed per cache file, so changing
the span silently changed the resolution per unit of time. That is not a
cosmetic issue: an under-resolved `f*` misses peaks, which biases it **low**,
which makes regret look **better** than it is — and it fails silently.

`assert_covers(env_start, env_end)` now runs before any compute is spent.
Without it `np.interp` clamps outside the cached span and reports a
plausible-looking but wrong regret; on `temperature`, `RBFInterpolator` would
extrapolate past the end of the sensor day and return confident nonsense.

### One bug found on the way

`--mi-clip-horizon` capped the MI criterion's lookahead at a hard-coded `1.0`.
That was only ever correct while the optimizer ran on a normalized `[0, 1]`
clock. On absolute environment time — Ackley starts at `t = -32` — it would
have clipped the horizon to nothing and disabled the flag without any error.
It now uses the run's actual `env_end`.

---

## Priority 2 — The log

`queries.csv` gained the queried point `x_0 … x_{d-1}`, its reading `y`, the
noise-free `true_value`, and four timestamps per iteration: `t_iter_start`,
`t_apply`, `t_result`, `t_update_done`.

This is what makes a run **re-scorable**. The headline metric for a system that
runs continuously is the time-average of `f*(t) - f(x_held(t), t)`, where
`x_held(t)` is the configuration actually in force — and no amount of
post-processing recovers that from regret snapshots alone.

**The convention:** `x_i` takes effect at `t_apply_i` (acquisition optimization
done, `f` being sampled) and holds until `t_apply_{i+1}`. While the optimizer
is choosing `x_i`, the *previous* configuration is still running. The opening
stretch `[0, t_apply_0)` is covered by a single `iteration = -1` row holding
the last observation of the initial design.

Supporting helpers in `common.py`: `load_objective()` rebuilds the objective
from `run.json` at the same oracle density the run used; `query_point()`
reassembles `x`; `queries_only()` drops the `iteration = -1` row.

---

## Verification

Three seeds × 60 s on both `ackley4d` and `temperature` (WSL), with an
independent checker asserting: replications split correctly, `env_time` is an
exact affine function of `t_iter_start`, stage timestamps monotone within and
across steps, `t_acq_fit == t_acq + t_fit`, `x` inside the domain, no query
*started* past `H`.

A throwaway scoring pass was then written against the log to confirm it is
sufficient. It rebuilds both objectives in one process, reconstructs
`x_held(t)` and integrates `R_time`:

- recomputed regret matches the logged `regret` column to **0.000e+00**;
- `R_time` differs from `avg_regret` by up to **32 %** on one temperature seed
  (0.373 → 0.492) — that seed chose good points but slowly, which is exactly
  what `avg_regret` cannot see.

---

## What this unlocks, against §3 of the note

| | Q1: which removal criterion is better? | Q2: which complete method runs better? |
|---|---|---|
| Clock, oracle, log | groundwork only | **sufficient** |
| Still missing | a fixed simulated-time query schedule; `max_dataset_size`; separating point *selection* from the removal *gate* | the `R_time` scorer |

Both questions are also still short three `CRITERIA` arms. The note asks for
six; the code has `wasserstein`, `mi`, `none`. **Oldest-point removal** and
**random removal** are the cheap ones and the decisive controls, especially for
Q1: if the Wasserstein criterion cannot beat random removal at the same data
budget, the paper's central claim does not hold. Dual Gate is the third.

---

## Compatibility notes

**Existing `queries.csv` files.** They still plot (`load_run` renames
`sim_time` → `env_time` and `t_response` → `t_acq_fit`), but they cannot be
re-scored — there is no `x` — and their numbers are not comparable, since their
environment speed was `span / duration_seconds`. Re-run anything you intend to
quote; do not relabel an old directory and carry it forward.

**Cached oracles.** Old `.npz` files are no longer read; the filename now
encodes span, density and grid resolution. They are not deleted. First run
rebuilds: a few seconds for `ackley4d`, ~30–60 s for `temperature`.

**Changed signatures.** `run_once` now returns `(log, info)` and requires
`env_t0` and `env_speed`. `criterion_settings(args, env_end)` takes the run's
end time. The `QUERY_FIELDS` constant became `query_fields(spatial_dim)`.
`load_run` splits replications wherever `iteration` stops increasing, not on
`iteration == 0`.

**The `iteration = -1` row.** Excluded from every metric inside the repo via
`queries_only`. It carries a real `regret` but `NaN` stage timings, so reading
`queries.csv` straight into pandas and calling `.mean()` gives wrong numbers
with no error. Filter on `iteration >= 0`.

**Environment.** These experiments run under the WSL venv
(`~/.venvs/wdbo_reproduce`); the compiled `wdbo_criterion` extension fails to load on the
Windows conda environment. Unrelated to these changes — see
[`../NOTE.md`](../NOTE.md).

---

## Known issue, pre-existing

`temperature` produced one negative regret out of 95 queries (−0.0016 against a
regret scale of ~1.0). Its oracle grid-searches space at 25×25 over `[0, 1]²`
while the optimizer may query anywhere, so it can occasionally find a point
better than the grid's best. This is the **spatial** grid, so
`--oracle-density` does not address it; it needs a higher
`oracle_grid_resolution` or an optimizer-based oracle.
