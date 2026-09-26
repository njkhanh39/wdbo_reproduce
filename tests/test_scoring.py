"""Analytic checks for the post-run scorer, with no GP dependencies."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from scoring import score_run


class MovingPeak:
    def assert_covers(self, start, end):
        assert 0 <= start <= end <= 1

    def oracle(self, t):
        return 0.0

    def evaluate(self, x, t):
        return -(x[0] - t) ** 2


class StaticPeak:
    def assert_covers(self, start, end):
        pass

    def oracle(self, t):
        return 0.0

    def evaluate(self, x, t):
        return -(x[0] - 1) ** 2


class SharpOracle:
    oracle_times = np.array([0.0, 0.25, 0.5, 1.0])

    def assert_covers(self, start, end):
        pass

    def oracle(self, t):
        return np.interp(t, self.oracle_times, [0, 1, 0, 0])

    def evaluate(self, x, t):
        return 0.0


def row(iteration, x, applied, env_time=0):
    return {"iteration": iteration, "x_0": x, "t_apply": applied,
            "env_time": env_time, "seed": 0, "regret": float("nan")}


class ScoringTests(unittest.TestCase):
    def test_moving_peak_converges_to_one_third(self):
        errors = []
        for n in (11, 101, 1001):
            result = score_run([row(-1, 0, 0)], MovingPeak(), 1, 0, 1, n)
            errors.append(abs(result["time_avg_regret"] - 1 / 3))
        self.assertTrue(errors[0] > errors[1] > errors[2])
        self.assertLess(errors[-1], 1e-6)

    def test_optimal_static_with_wait_and_no_queries(self):
        for horizon in (1, 10):
            result = score_run([row(-1, 1, 0)], StaticPeak(), horizon, 0, 0)
            self.assertEqual(result["time_avg_regret"], 0)

    def test_switch_between_grid_nodes_and_overrun(self):
        run = [row(-1, 0, 0), row(0, 1, 0.37), row(1, 1, 0.9)]
        run[-1]["wall_time"] = 1.2  # final fit/clean may overrun H
        result = score_run(run, StaticPeak(), 1, 0, 0, 3)
        self.assertAlmostEqual(result["time_avg_regret"], 0.37)
        self.assertEqual(run[1]["regret"], 0)
        self.assertEqual(run[2]["regret"], 0)

    def test_old_log_rejected(self):
        with self.assertRaisesRegex(ValueError, "initial held"):
            score_run([row(0, 0, 0)], MovingPeak(), 1, 0, 1)
        with self.assertRaisesRegex(ValueError, "application times"):
            score_run([{"iteration": -1, "x_0": 0}], MovingPeak(), 1, 0, 1)
        with self.assertRaisesRegex(ValueError, "measurement time"):
            score_run([row(-1, 0, 0), row(0, 0, 0.5, 0.1)], MovingPeak(), 1, 0, 1)

    def test_oracle_table_knots_prevent_aliasing(self):
        result = score_run([row(-1, 0, 0)], SharpOracle(), 1, 0, 1, 2)
        self.assertAlmostEqual(result["time_avg_regret"], 0.25)


if __name__ == "__main__":
    unittest.main()
