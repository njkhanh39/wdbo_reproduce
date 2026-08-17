import pytest
import torch
from botorch.utils.sampling import draw_sobol_samples

from wdbo_algo.benchmarks import (
	BENCHMARKS,
	ackley,
	eggholder,
	hartmann3,
	hartmann6,
	powell,
	rastrigin,
	rosenbrock,
	schwefel,
	shekel,
	styblinski_tang,
)


DTYPE = torch.double


@pytest.mark.parametrize(
	("name", "spatial_dimension", "lower", "upper"),
	[
		("rastrigin", 4, -4.0, 4.0),
		("schwefel", 3, -500.0, 500.0),
		("styblinski_tang", 3, -5.0, 5.0),
		("eggholder", 1, -512.0, 512.0),
		("ackley", 3, -32.0, 32.0),
		("rosenbrock", 2, -1.0, 1.5),
		("shekel", 3, 0.0, 10.0),
		("hartmann3", 2, 0.0, 1.0),
		("hartmann6", 5, 0.0, 1.0),
		("powell", 3, -4.0, 5.0),
	],
)
def test_registry_matches_appendix_h2(name, spatial_dimension, lower, upper):
	benchmark = BENCHMARKS[name]
	assert benchmark.spatial_dimension == spatial_dimension
	assert benchmark.dim == spatial_dimension + 1
	assert torch.equal(
		benchmark.bounds,
		torch.tensor(
			[[lower] * benchmark.dim, [upper] * benchmark.dim], dtype=DTYPE
		),
	)


def test_reference_values_match_standard_paper_formulas():
	assert ackley(torch.zeros(4, dtype=DTYPE)).item() == pytest.approx(0.0, abs=1e-12)
	assert rastrigin(torch.zeros(5, dtype=DTYPE)).item() == 0.0
	assert schwefel(torch.full((4,), 420.968746, dtype=DTYPE)).item() == pytest.approx(
		0.0, abs=5e-4
	)
	assert styblinski_tang(
		torch.full((4,), -2.903534, dtype=DTYPE)
	).item() == pytest.approx(-39.16599 * 4, abs=1e-3)
	assert eggholder(torch.tensor([512.0, 404.2319], dtype=DTYPE)).item() == pytest.approx(
		-959.6407, abs=1e-3
	)
	assert rosenbrock(torch.ones(3, dtype=DTYPE)).item() == 0.0
	assert shekel(torch.full((4,), 4.0, dtype=DTYPE)).item() == pytest.approx(
		-10.5364, abs=1e-3
	)
	assert hartmann3(
		torch.tensor([0.114614, 0.555649, 0.852547], dtype=DTYPE)
	).item() == pytest.approx(-3.86278, abs=1e-4)
	assert hartmann6(
		torch.tensor(
			[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573],
			dtype=DTYPE,
		)
	).item() == pytest.approx(-3.32237, abs=1e-4)
	assert powell(torch.zeros(4, dtype=DTYPE)).item() == 0.0


def test_vectorized_evaluation_preserves_dtype_and_gradients():
	points = torch.rand(7, 3, dtype=DTYPE, requires_grad=True)
	values = BENCHMARKS["rosenbrock"].evaluate(points[..., :2], points[..., 2])
	assert values.shape == torch.Size([7])
	assert values.dtype == DTYPE
	values.sum().backward()
	assert points.grad is not None
	assert torch.isfinite(points.grad).all()


def test_botorch_native_bounds_and_q_batch_calling_convention():
	benchmark = BENCHMARKS["hartmann6"]
	points = draw_sobol_samples(bounds=benchmark.bounds, n=3, q=2, seed=7)
	values = benchmark(points)
	assert points.shape == torch.Size([3, 2, benchmark.dim])
	assert values.shape == torch.Size([3, 2])
	assert torch.isfinite(values).all()


def test_objective_is_negated_loss_for_ucb_maximization():
	benchmark = BENCHMARKS["ackley"]
	x = torch.rand(5, benchmark.spatial_dimension, dtype=DTYPE)
	t = torch.linspace(0.0, 1.0, 5, dtype=DTYPE)
	assert torch.equal(benchmark.objective(x, t), -benchmark.evaluate(x, t))


def test_normalized_mapping_includes_time_as_last_native_coordinate():
	benchmark = BENCHMARKS["rastrigin"]
	x = torch.tensor([[0.0] * 4, [1.0] * 4], dtype=DTYPE)
	t = torch.tensor([0.25, 0.75], dtype=DTYPE)
	z = torch.cat((benchmark.to_native_space(x), benchmark.temporal_coordinate(t, like=x)[:, None]), dim=-1)
	assert torch.equal(benchmark.evaluate(x, t), benchmark(z))
	assert torch.equal(z[0], torch.tensor([-4.0, -4.0, -4.0, -4.0, -2.0], dtype=DTYPE))
	assert torch.equal(z[1], torch.tensor([4.0, 4.0, 4.0, 4.0, 2.0], dtype=DTYPE))


def test_optimum_oracle_is_deterministic_and_no_worse_than_center():
	benchmark = BENCHMARKS["hartmann3"]
	center = torch.full((benchmark.spatial_dimension,), 0.5, dtype=DTYPE)
	value = benchmark.optimum_value(0.4, n_candidates=256)
	assert torch.equal(value, benchmark.optimum_value(0.4, n_candidates=256))
	assert value <= benchmark.evaluate(center, 0.4)


@pytest.mark.parametrize(
	("function", "bad_dimension"),
	[(eggholder, 3), (shekel, 3), (hartmann3, 4), (hartmann6, 5), (powell, 5)],
)
def test_dimension_validation(function, bad_dimension):
	with pytest.raises(ValueError):
		function(torch.zeros(bad_dimension, dtype=DTYPE))
