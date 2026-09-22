"""Post-run regret scoring; deliberately independent of the optimizer stack."""
from __future__ import annotations

import numpy as np


def score_run(run: list[dict], objective, duration_seconds: float,
              env_start: float, env_speed: float, grid_points: int = 200) -> dict:
    """Integrate noise-free regret of the configuration actually held on [0, H].

    Every application event is an integration boundary, even when it falls
    between common grid points. Thus a switch never gets smeared over a cell.
    Increasing grid_points tests convergence of the trapezoid rule on each
    smooth segment.
    """
    if not np.isfinite(duration_seconds) or duration_seconds <= 0 or grid_points < 2:
        raise ValueError("Scoring requires H > 0 and at least two grid points")
    if not np.isfinite(env_start) or not np.isfinite(env_speed) or env_speed < 0:
        raise ValueError("Environment clock must be finite and nondecreasing")
    if not run or run[0].get("iteration") != -1:
        raise ValueError("Cannot score log without the initial held configuration; rerun it")
    columns = sorted(k for k in run[0] if k.startswith("x_") and k[2:].isdigit())
    if not columns or columns != [f"x_{i}" for i in range(len(columns))]:
        raise ValueError("Cannot score log without complete x coordinates; rerun it")
    if any(not all(k in row for k in columns) for row in run):
        raise ValueError("Cannot score log with missing x coordinates; rerun it")
    points = [np.asarray([row[k] for k in columns], dtype=float) for row in run]
    if any(not np.all(np.isfinite(point)) for point in points):
        raise ValueError("Logged configurations must be finite")
    try:
        applications = np.asarray([row["t_apply"] for row in run], dtype=float)
    except KeyError as exc:
        raise ValueError("Cannot score log without application times; rerun it") from exc
    if not np.all(np.isfinite(applications)) or applications[0] != 0 or np.any(np.diff(applications) < 0):
        raise ValueError("Application times must be finite, begin at zero and be ordered")
    if any(row["iteration"] != i - 1 for i, row in enumerate(run)):
        raise ValueError("Run rows must have consecutive iterations starting at -1")
    for row, applied in zip(run[1:], applications[1:]):
        if applied >= duration_seconds:
            raise ValueError("A query was applied after the deadline; rerun it")
        expected = env_start + env_speed * applied
        if not np.isclose(float(row["env_time"]), expected, rtol=1e-9, atol=1e-8):
            raise ValueError("Query measurement time disagrees with t_apply; rerun it")
    objective.assert_covers(env_start, env_start + env_speed * duration_seconds)

    grid = np.linspace(0.0, duration_seconds, grid_points)
    events = applications[(applications > 0) & (applications < duration_seconds)]
    # Cached oracles are piecewise linear between their own time samples.
    # Include those samples so a fast-moving benchmark is not aliased by the
    # coarser grid used only for plotting the running average.
    oracle_knots = np.empty(0)
    if env_speed > 0 and hasattr(objective, "oracle_times"):
        oracle_knots = (np.asarray(objective.oracle_times, dtype=float) - env_start) / env_speed
        oracle_knots = oracle_knots[(oracle_knots > 0) & (oracle_knots < duration_seconds)]
    knots = np.unique(np.concatenate((grid, events, oracle_knots)))
    areas = np.zeros(len(knots))
    for i, (left, right) in enumerate(zip(knots[:-1], knots[1:])):
        held = points[np.searchsorted(applications, (left + right) / 2, side="right") - 1]
        t_left = env_start + env_speed * left
        t_right = env_start + env_speed * right
        r_left = objective.oracle(t_left) - objective.evaluate(held, t_left)
        r_right = objective.oracle(t_right) - objective.evaluate(held, t_right)
        areas[i + 1] = areas[i] + (right - left) * (r_left + r_right) / 2
    cumulative = areas[np.searchsorted(knots, grid)]
    curve = np.empty_like(grid)
    curve[0] = objective.oracle(env_start) - objective.evaluate(points[0], env_start)
    curve[1:] = cumulative[1:] / grid[1:]

    # Query regret is a distinct diagnostic at each logged measurement time.
    for row, point in zip(run[1:], points[1:]):
        t_env = float(row["env_time"])
        row["regret"] = float(objective.oracle(t_env) - objective.evaluate(point, t_env))
    return {"grid": grid, "running_time_regret": curve,
            "time_avg_regret": float(cumulative[-1] / duration_seconds)}
