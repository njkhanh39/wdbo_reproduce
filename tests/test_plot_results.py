import csv

import pytest

from wdbo_algo.plot_results import load_run, plot_runs


FIELDS = [
	"benchmark",
	"method",
	"seed",
	"duration_seconds",
	"phase",
	"iteration",
	"normalized_time",
	"loss",
	"objective",
	"observed_objective",
	"optimum_loss",
	"instantaneous_regret",
	"response_seconds",
	"dataset_size",
	"spatial_x",
]


def _write_run(path, seed, response_offset=0.0):
	rows = []
	for iteration, (time, regret, response, size) in enumerate(
		[(0.1, 4.0, 0.5, 15), (0.5, 2.0, 1.0, 20), (0.9, 1.0, 1.5, 12)]
	):
		rows.append(
			{
				"benchmark": "ackley",
				"method": "W-DBO",
				"seed": seed,
				"duration_seconds": 600,
				"phase": "optimization",
				"iteration": iteration,
				"normalized_time": time,
				"loss": regret,
				"objective": -regret,
				"observed_objective": -regret,
				"optimum_loss": 0,
				"instantaneous_regret": regret,
				"response_seconds": response + response_offset,
				"dataset_size": size,
				"spatial_x": "0 0 0",
			}
		)
	with path.open("w", newline="", encoding="utf-8") as stream:
		writer = csv.DictWriter(stream, fieldnames=FIELDS)
		writer.writeheader()
		writer.writerows(rows)


def test_load_run_computes_replication_averages(tmp_path):
	path = tmp_path / "ackley_wdbo_seed3.csv"
	_write_run(path, seed=3)
	run = load_run(path)
	assert run.benchmark == "ackley"
	assert run.method == "W-DBO"
	assert run.seed == 3
	assert run.average_response == pytest.approx(1.0)
	assert run.average_regret == pytest.approx(7.0 / 3.0)
	assert run.times.tolist() == pytest.approx([60.0, 300.0, 540.0])


def test_plot_runs_creates_paper_style_figure(tmp_path):
	first = tmp_path / "ackley_wdbo_seed0.csv"
	second = tmp_path / "ackley_wdbo_seed1.csv"
	_write_run(first, seed=0)
	_write_run(second, seed=1, response_offset=0.25)
	output = tmp_path / "ackley.png"
	created, summary = plot_runs([first, second], output=output, grid_points=21)
	assert created == output.resolve()
	assert output.stat().st_size > 10_000
	assert summary["W-DBO"]["runs"] == 2
