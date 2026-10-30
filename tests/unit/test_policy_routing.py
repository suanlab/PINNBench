"""Tests for budget-aware policy routing helpers."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _early_predictor_module() -> ModuleType:
    return importlib.import_module("causal_pinn.training.early_predictor")


def _policy_routing_module() -> ModuleType:
    return importlib.import_module("causal_pinn.training.policy_routing")


def _make_sample(
    *,
    pde: str,
    policy: str,
    seed: int,
    target: float,
    physics_final: float,
    physics_slope: float,
    total_final: float,
) -> object:
    PredictorSample = _early_predictor_module().PredictorSample

    return PredictorSample(
        features={
            "physics_loss_final": physics_final,
            "physics_loss_slope": physics_slope,
            "total_final": total_final,
        },
        target=target,
        pde=pde,
        policy=policy,
        seed=seed,
    )


class TestBudgetMath:
    def test_budget_ratio_and_savings(self) -> None:
        policy_routing = _policy_routing_module()
        budget_ratio = policy_routing.budget_ratio
        budget_savings = policy_routing.budget_savings

        ratio = budget_ratio(3, 10, 500)
        savings = budget_savings(3, 10, 500)

        assert ratio == pytest.approx((3 * 10 + 490) / (3 * 500))
        assert savings == pytest.approx(1.0 - ratio)

    def test_budget_ratio_rejects_invalid_probe(self) -> None:
        budget_ratio = _policy_routing_module().budget_ratio

        with pytest.raises(ValueError):
            budget_ratio(3, 600, 500)


class TestNormalization:
    def test_normalize_policy_name(self) -> None:
        normalize_policy_name = _policy_routing_module().normalize_policy_name

        assert normalize_policy_name("hybrid") == "hybrid_full"
        assert normalize_policy_name("pinn_only") == "pinn_only"

    def test_normalize_pde_group_strict(self) -> None:
        normalize_pde_group = _policy_routing_module().normalize_pde_group

        assert normalize_pde_group("heat1d_pino", strict=True) == "heat1d"
        assert normalize_pde_group("wave2d", strict=True) == "wave2d"
        assert normalize_pde_group("heat1d_pino", strict=False) == "heat1d_pino"


class TestRoutingHelpers:
    def test_policy_mean_values(self) -> None:
        import numpy as np

        policy_mean_values = _policy_routing_module().policy_mean_values

        samples = [
            _make_sample(
                pde="a",
                policy="hybrid",
                seed=1,
                target=0.0,
                physics_final=0.1,
                physics_slope=-0.1,
                total_final=0.2,
            ),
            _make_sample(
                pde="a",
                policy="hybrid_full",
                seed=2,
                target=0.0,
                physics_final=0.2,
                physics_slope=-0.2,
                total_final=0.3,
            ),
        ]

        means = policy_mean_values(samples, np.array([0.04, 0.02]))
        assert means == {"hybrid_full": pytest.approx(0.03)}

    def test_training_prior_filters_to_candidate_set(self) -> None:
        import numpy as np

        train_policy_prior = _policy_routing_module().train_policy_prior

        samples = [
            _make_sample(
                pde="a",
                policy="pinn_only",
                seed=1,
                target=0.04,
                physics_final=0.4,
                physics_slope=-0.01,
                total_final=0.5,
            ),
            _make_sample(
                pde="a",
                policy="hybrid_full",
                seed=2,
                target=0.02,
                physics_final=0.2,
                physics_slope=-0.03,
                total_final=0.3,
            ),
        ]

        selected = train_policy_prior(
            samples,
            np.array([0.04, 0.02]),
            {"pinn_only"},
        )
        assert selected == "pinn_only"

    def test_heuristic_policy_choice(self) -> None:
        heuristic_policy_choice = _policy_routing_module().heuristic_policy_choice

        samples = [
            _make_sample(
                pde="a",
                policy="pinn_only",
                seed=1,
                target=0.04,
                physics_final=0.4,
                physics_slope=-0.01,
                total_final=0.5,
            ),
            _make_sample(
                pde="a",
                policy="hybrid_full",
                seed=1,
                target=0.02,
                physics_final=0.2,
                physics_slope=-0.03,
                total_final=0.3,
            ),
        ]

        assert heuristic_policy_choice(samples, "physics_loss_final") == "hybrid_full"

    def test_fallback_policy_choice(self) -> None:
        fallback_policy_choice = _policy_routing_module().fallback_policy_choice

        selected, used = fallback_policy_choice(
            {
                "hybrid_full": (0.10, 0.03),
                "pinn_only": (0.11, 0.01),
            },
            fallback_policy="pinn_only",
            margin_std_ratio=1.0,
        )

        assert selected == "pinn_only"
        assert used is True


class TestRoutingEvaluation:
    def test_remap_dataset_groups(self) -> None:
        PredictorDataset = _early_predictor_module().PredictorDataset
        remap_dataset_groups = _policy_routing_module().remap_dataset_groups

        dataset = PredictorDataset(
            samples=[
                _make_sample(
                    pde="heat1d_pino",
                    policy="pinn_only",
                    seed=1,
                    target=0.04,
                    physics_final=0.4,
                    physics_slope=-0.01,
                    total_final=0.5,
                )
            ]
        )
        remapped = remap_dataset_groups(dataset, strict=True)

        assert remapped.samples[0].pde == "heat1d"

    def test_evaluate_policy_routing(self) -> None:
        PredictorDataset = _early_predictor_module().PredictorDataset
        evaluate_policy_routing = _policy_routing_module().evaluate_policy_routing

        dataset = PredictorDataset(
            samples=[
                _make_sample(
                    pde="heat1d",
                    policy="pinn_only",
                    seed=1,
                    target=0.40,
                    physics_final=0.40,
                    physics_slope=-0.01,
                    total_final=0.50,
                ),
                _make_sample(
                    pde="heat1d",
                    policy="hybrid_full",
                    seed=1,
                    target=0.10,
                    physics_final=0.10,
                    physics_slope=-0.05,
                    total_final=0.20,
                ),
                _make_sample(
                    pde="wave2d",
                    policy="pinn_only",
                    seed=1,
                    target=0.50,
                    physics_final=0.50,
                    physics_slope=-0.01,
                    total_final=0.60,
                ),
                _make_sample(
                    pde="wave2d",
                    policy="hybrid_full",
                    seed=1,
                    target=0.12,
                    physics_final=0.12,
                    physics_slope=-0.06,
                    total_final=0.22,
                ),
                _make_sample(
                    pde="kdv",
                    policy="pinn_only",
                    seed=1,
                    target=0.30,
                    physics_final=0.30,
                    physics_slope=-0.02,
                    total_final=0.40,
                ),
                _make_sample(
                    pde="kdv",
                    policy="hybrid_full",
                    seed=1,
                    target=0.08,
                    physics_final=0.08,
                    physics_slope=-0.07,
                    total_final=0.18,
                ),
            ]
        )

        summary = evaluate_policy_routing(
            dataset,
            probe_epochs=10,
            total_epochs=500,
            strict_grouping=False,
            margin_std_ratio=1.0,
        )

        selectors = summary["selectors"]
        assert summary["groups"] == ["heat1d", "wave2d", "kdv"]
        assert selectors["physics_final"]["selection_accuracy"] == pytest.approx(1.0)
        assert selectors["physics_slope"]["selection_accuracy"] == pytest.approx(1.0)
        assert selectors["random_forest"]["mean_budget_savings"] == pytest.approx(
            1.0 - ((2 * 10 + 490) / (2 * 500))
        )
        assert selectors["random_expected"]["selection_accuracy"] == pytest.approx(0.5)
