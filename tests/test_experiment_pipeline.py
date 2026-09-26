"""Timed-loop contract checks using a deterministic fake optimizer and clock."""
import sys
import copy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import common
from scoring import score_run
from wdbo_algo.candidate_optimizer import CandidatePruningOptimizer


class Clock:
    now = 0.0

    def read(self):
        return self.now


class Objective:
    noise_std = 0.0

    def __init__(self, clock):
        self.clock = clock
        self.allow_oracle = False

    def evaluate(self, x, t):
        self.clock.now += 0.1
        return -float((x[0] - t) ** 2)

    def oracle(self, t):
        if not self.allow_oracle:
            raise AssertionError("oracle called during timed optimization")
        return 0.0

    def assert_covers(self, start, end):
        pass


class Optimizer:
    _lambda = _lS = _lT = _noise = _budget = 1.0
    _last_min_criterion = _last_min_lT = 1.0
    _budget_spent = 0.0

    def __init__(self, clock, acquisition_seconds=0.2):
        self.clock = clock
        self.acquisition_seconds = acquisition_seconds
        self.count = 0
        self.clean_calls = 0

    def next_query(self, t):
        self.clock.now += self.acquisition_seconds
        return np.array([0.5])

    def tell(self, x, t, y):
        self.clock.now += 0.3
        self.count += 1

    def clean(self, t):
        self.clean_calls += 1
        self.clock.now += 0.1

    def dataset_size(self):
        return self.count


def run_with_clock(duration, acquisition_seconds=0.2):
    clock = Clock()
    objective = Objective(clock)
    optimizer = Optimizer(clock, acquisition_seconds)
    with patch.object(common, "build_optimizer", return_value=optimizer), \
         patch.object(common.time, "perf_counter", side_effect=clock.read), \
         patch.object(common, "print_progress"):
        log, info = common.run_once(objective, duration, 2, 0.25, 0,
                                    0.0, 1.0, criterion="none")
    return log, info, objective, optimizer


def test_none_uses_application_time_without_oracle_or_clean():
    log, info, objective, optimizer = run_with_clock(1.0)
    assert optimizer.clean_calls == 0
    assert len(log) == 3  # initial configuration plus two real queries
    assert info["overran"]
    for row in log[1:]:
        assert row["env_time"] == pytest.approx(info["env_start"] + row["t_apply"])
        assert np.isnan(row["regret"])
    objective.allow_oracle = True
    score = score_run(log, objective, 1.0, info["env_start"], 1.0)
    assert np.isfinite(score["time_avg_regret"])
    assert all(np.isfinite(row["regret"]) for row in log[1:])


def test_acquisition_past_deadline_issues_no_query():
    log, info, objective, optimizer = run_with_clock(0.01, acquisition_seconds=0.2)
    assert len(log) == 1
    assert info["overran"]
    assert optimizer.clean_calls == 0
    objective.allow_oracle = True
    assert np.isfinite(score_run(log, objective, 0.01, info["env_start"], 1.0)["time_avg_regret"])


def test_same_seed_log_roundtrip_and_rescore(tmp_path):
    log, info, objective, _ = run_with_clock(0.01)
    objective.allow_oracle = True
    metadata = {"env_schedule": info, "args": {"duration_seconds": 0.01}}
    per_seed, _ = common.save_run(tmp_path, [log, copy.deepcopy(log)], metadata,
                                  0.01, objective, [info, info])
    loaded, restored_metadata = common.load_run(tmp_path)
    assert len(loaded) == 2
    assert all(len(run) == 1 and run[0]["seed"] == 0 for run in loaded)
    rescored, _ = common.score_results(loaded, restored_metadata, objective, 0.01)
    assert [row["time_avg_regret"] for row in rescored] == pytest.approx(
        [row["time_avg_regret"] for row in per_seed])


def test_dual_gate_uses_shared_dataset_floor():
    objective = type("Objective", (), {"spatial_domain": np.array([[0.0, 1.0]])})()
    optimizer = common.build_optimizer(
        objective, n_initial_observations=5, alpha=0.25,
        criterion="dual_gate", min_dataset_size=5,
        mi_options={"candidate_pool_size": 16}, seed=0)
    assert isinstance(optimizer, CandidatePruningOptimizer)
    assert optimizer._min_dataset_size == 5
    assert optimizer._candidate_pool_size == 16
