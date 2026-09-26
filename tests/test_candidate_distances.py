import numpy as np
import pytest
import gpytorch

from wdbo_algo.candidate_model import learn_model_space_time
from wdbo_algo.candidate_optimizer import (
	CandidatePruningOptimizer,
    dynamic_deletion_limit,
    effective_dimension,
    gaussian_hellinger,
    joint_gate_load,
    jensen_shannon_choice_distance,
    robust_scale,
    single_prune_cost,
    update_single_prune_credit,
)


def test_robust_scale_is_positive_and_affine_equivariant():
    values = np.array([-10.0, 0.0, 1.0, 2.0, 1000.0])
    base = robust_scale(values)
    assert base > 0
    assert robust_scale(7.0 * values + 19.0) == pytest.approx(7.0 * base)
    assert robust_scale(np.ones(5)) > 0


def test_gaussian_hellinger_identity_bounds_and_affine_invariance():
    mean_a = np.array([0.0, 1.0, -2.0])
    std_a = np.array([1.0, 0.5, 3.0])
    assert np.allclose(gaussian_hellinger(mean_a, std_a, mean_a, std_a), 0.0)

    mean_b = np.array([0.5, -1.0, 2.0])
    std_b = np.array([2.0, 0.7, 1.0])
    distance = gaussian_hellinger(mean_a, std_a, mean_b, std_b)
    transformed = gaussian_hellinger(
        4.0 * mean_a - 8.0,
        4.0 * std_a,
        4.0 * mean_b - 8.0,
        4.0 * std_b,
    )
    assert np.all((0.0 <= distance) & (distance <= 1.0))
    assert transformed == pytest.approx(distance)


def test_js_choice_distance_is_symmetric_bounded_and_affine_invariant():
    first = np.array([0.0, 1.0, 3.0, 2.0])
    second = np.array([1.0, 1.5, 2.2, 2.1])
    distance = jensen_shannon_choice_distance(first, second, temperature=0.8)
    assert 0.0 <= distance <= 1.0
    assert jensen_shannon_choice_distance(first, first) == pytest.approx(0.0)
    assert jensen_shannon_choice_distance(second, first, temperature=0.8) == pytest.approx(distance)
    assert jensen_shannon_choice_distance(
        5.0 * first - 11.0,
        5.0 * second - 11.0,
        temperature=0.8,
    ) == pytest.approx(distance)


def test_js_choice_distance_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        jensen_shannon_choice_distance([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        jensen_shannon_choice_distance([1.0], [1.0], temperature=0.0)


def test_dynamic_deletion_limit_is_sublinear_and_fraction_bounded():
    assert dynamic_deletion_limit(2) == 1
    assert dynamic_deletion_limit(15) == 4
    assert dynamic_deletion_limit(100) == 10
    assert dynamic_deletion_limit(10_000) == 100
    with pytest.raises(ValueError):
        dynamic_deletion_limit(0)
    with pytest.raises(ValueError):
        dynamic_deletion_limit(10, 1.1)


def test_joint_gate_load_uses_the_tighter_normalized_gate():
    load = joint_gate_load(
        nas=np.array([0.1, 0.3]),
        normalized_margin=1.0,
        hellinger=np.array([0.08, 0.02]),
        hellinger_threshold=0.1,
    )
    assert load == pytest.approx([0.8, 0.6])
    assert np.all(load < 1.0)


def test_effective_dimension_and_dynamic_single_prune_budget():
    assert effective_dimension([4.0, 1.0, 0.0], noise=1.0) == pytest.approx(1.3)
    assert single_prune_cost(0.0, base_cost=0.1) == pytest.approx(0.1)
    assert single_prune_cost(1.0, base_cost=0.1) == pytest.approx(1.0)
    credit, pressure, refill = update_single_prune_credit(
        credit=0.2,
        elapsed_time=0.1,
        temporal_lengthscale=0.2,
        alpha=0.25,
        data_size=20,
        reserve_size=10,
        credit_cap=1.0,
    )
    assert pressure == pytest.approx(1.0)
    assert refill == pytest.approx(0.25)
    assert credit == pytest.approx(0.45)


def test_dynamic_budget_clean_can_delete_multiple_points_sequentially(monkeypatch):
    optimizer = CandidatePruningOptimizer(
        np.array([[0.0, 1.0], [0.0, 1.0]]),
        gpytorch.kernels.MaternKernel,
        gpytorch.kernels.MaternKernel,
        pruning_method="dual_gate_budget",
    )
    optimizer._xx_tt = np.zeros((12, 3))
    optimizer._yy = np.arange(12.0)
    optimizer._current_time = 0.0
    optimizer._lT = 0.1
    optimizer._single_prune_credit = 0.5
    monkeypatch.setattr(optimizer, "_effective_dimension_and_reserve", lambda: (5.0, 10))
    monkeypatch.setattr(
        optimizer,
        "_candidate_relevance",
        lambda _t: (
            np.linspace(0.1, 0.9, len(optimizer._yy)),
            np.ones(len(optimizer._yy), dtype=bool),
            {},
        ),
    )
    monkeypatch.setattr(optimizer, "update_surrogate_model", lambda verbose=False: None)

    optimizer._clean_sequential_with_dynamic_budget(0.1)

    assert len(optimizer._yy) == 10
    assert optimizer.last_clean_diagnostics["deleted"] == 2
    assert optimizer.last_clean_diagnostics["stopped"] == "dataset_reserve"
    assert optimizer.last_clean_diagnostics["sequential_spend"] > 0


def test_dynamic_budget_clean_respects_dataset_reserve(monkeypatch):
    optimizer = CandidatePruningOptimizer(
        np.array([[0.0, 1.0], [0.0, 1.0]]),
        gpytorch.kernels.MaternKernel,
        gpytorch.kernels.MaternKernel,
        pruning_method="dual_gate_budget",
    )
    optimizer._xx_tt = np.zeros((10, 3))
    optimizer._yy = np.arange(10.0)
    optimizer._current_time = 0.0
    optimizer._lT = 0.1
    monkeypatch.setattr(optimizer, "_effective_dimension_and_reserve", lambda: (7.5, 10))

    optimizer._clean_sequential_with_dynamic_budget(0.1)

    assert len(optimizer._yy) == 10
    assert optimizer.last_clean_diagnostics["deleted"] == 0
    assert optimizer.last_clean_diagnostics["stopped"] == "dataset_reserve"


def test_analytic_leave_one_out_matches_explicit_reduced_gp():
	train_x = np.array([
		[0.1, 0.2, 0.0], [0.8, 0.3, 0.1],
		[0.4, 0.9, 0.2], [0.7, 0.7, 0.3],
	])
	train_y = np.array([-1.0, 0.4, 0.7, -0.1])
	test_x = np.array([[0.2, 0.6, 0.35], [0.9, 0.1, 0.35], [0.5, 0.5, 0.35]])
	model_args = (
		gpytorch.kernels.MaternKernel, [2.5],
		gpytorch.kernels.MaternKernel, [1.5],
	)
	full_model = learn_model_space_time(train_x, *model_args, train_y, fit_model=False).double()
	full_model.mean_module.constant = 0.17
	full_model.covar_module.outputscale = 1.3
	full_model.covar_module.base_kernel.kernels[0].lengthscale = 0.6
	full_model.covar_module.base_kernel.kernels[1].lengthscale = 0.25
	full_model.likelihood.noise = 0.08
	full_model.eval()
	full_model.likelihood.eval()

	optimizer = CandidatePruningOptimizer(
		np.array([[0.0, 1.0], [0.0, 1.0]]),
		gpytorch.kernels.MaternKernel,
		gpytorch.kernels.MaternKernel,
		spatial_kernel_args=[2.5], temporal_kernel_args=[1.5],
	)
	optimizer._xx_tt = train_x
	optimizer._yy_normalized = train_y
	optimizer._gpr = full_model
	full_mean, full_std = optimizer._predictive_stats(full_model, test_x)
	loo_mean, loo_std = optimizer._leave_one_out_predictive_stats(test_x, full_mean, full_std)

	for index in range(len(train_x)):
		keep = np.arange(len(train_x)) != index
		reduced = learn_model_space_time(
			train_x[keep], *model_args, train_y[keep], fit_model=False
		).double()
		reduced.load_state_dict(full_model.state_dict())
		reduced.eval()
		reduced.likelihood.eval()
		expected_mean, expected_std = optimizer._predictive_stats(reduced, test_x)
		assert loo_mean[:, index] == pytest.approx(expected_mean, abs=1e-12)
		assert loo_std[:, index] == pytest.approx(expected_std, abs=1e-12)
