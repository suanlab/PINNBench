"""Tests for early-training predictor module."""

from __future__ import annotations

import math

import numpy as np
import pytest


def _make_stage2_losses(n_epochs: int = 100, with_data: bool = True) -> list[dict]:
    losses = []
    for i in range(n_epochs):
        ep: dict = {
            "total": 0.5 * math.exp(-i / 50.0) + 0.01,
            "physics_loss": 0.3 * math.exp(-i / 60.0) + 0.005,
            "bc_loss": 0.1 * math.exp(-i / 40.0) + 0.002,
            "lambda_physics": 0.1 + 0.9 * min(1.0, i / 250.0),
        }
        if with_data:
            ep["data_loss"] = 0.1 * math.exp(-i / 30.0) + 0.003
        losses.append(ep)
    return losses


class TestFeatureExtractor:
    def test_extract_basic_features(self) -> None:
        from causal_pinn.training.early_predictor import FeatureExtractor

        ext = FeatureExtractor(window_size=50)
        losses = _make_stage2_losses(100)
        features = ext.extract(losses)

        assert len(features) > 0
        assert "total_initial" in features
        assert "total_final" in features
        assert "total_mean" in features
        assert "total_slope" in features
        assert "total_ratio" in features
        assert "physics_loss_std" in features
        assert "bc_loss_min" in features

    def test_extract_with_data_loss(self) -> None:
        from causal_pinn.training.early_predictor import FeatureExtractor

        ext = FeatureExtractor(window_size=50)
        losses = _make_stage2_losses(100, with_data=True)
        features = ext.extract(losses)

        assert "data_loss_initial" in features
        assert "data_loss_slope" in features

    def test_extract_without_data_loss(self) -> None:
        from causal_pinn.training.early_predictor import FeatureExtractor

        ext = FeatureExtractor(window_size=50)
        losses = _make_stage2_losses(100, with_data=False)
        features = ext.extract(losses)

        assert "data_loss_initial" not in features

    def test_extract_empty_history(self) -> None:
        from causal_pinn.training.early_predictor import FeatureExtractor

        ext = FeatureExtractor(window_size=50)
        features = ext.extract([])
        assert features == {}

    def test_extract_window_smaller_than_history(self) -> None:
        from causal_pinn.training.early_predictor import FeatureExtractor

        ext = FeatureExtractor(window_size=10)
        losses = _make_stage2_losses(100)
        features = ext.extract(losses)

        assert features["total_initial"] == pytest.approx(losses[0]["total"], abs=1e-6)

    def test_extract_window_larger_than_history(self) -> None:
        from causal_pinn.training.early_predictor import FeatureExtractor

        ext = FeatureExtractor(window_size=200)
        losses = _make_stage2_losses(100)
        features = ext.extract(losses)

        assert len(features) > 0
        assert features["total_initial"] == pytest.approx(losses[0]["total"], abs=1e-6)
        assert features["total_final"] == pytest.approx(losses[-1]["total"], abs=1e-6)

    def test_extract_ratio_features(self) -> None:
        from causal_pinn.training.early_predictor import FeatureExtractor

        ext = FeatureExtractor(window_size=50)
        losses = _make_stage2_losses(100, with_data=True)
        features = ext.extract(losses)

        assert "physics_loss_data_loss_ratio_final" in features
        assert "bc_loss_total_ratio_final" in features

    def test_feature_names(self) -> None:
        from causal_pinn.training.early_predictor import FeatureExtractor

        ext = FeatureExtractor(window_size=50)
        names = ext.feature_names
        assert len(names) > 0
        assert "total_initial" in names
        assert "physics_loss_slope" in names

    def test_n_features(self) -> None:
        from causal_pinn.training.early_predictor import FeatureExtractor

        ext = FeatureExtractor(window_size=50)
        assert ext.n_features == len(ext.feature_names)


class TestPredictorSampleAndDataset:
    def test_predictor_sample_creation(self) -> None:
        from causal_pinn.training.early_predictor import PredictorSample

        sample = PredictorSample(
            features={"a": 1.0, "b": 2.0},
            target=0.05,
            pde="test_pde",
            policy="hybrid",
            seed=42,
        )
        assert sample.pde == "test_pde"
        assert sample.target == 0.05

    def test_dataset_feature_matrix(self) -> None:
        from causal_pinn.training.early_predictor import (
            PredictorDataset,
            PredictorSample,
        )

        samples = [
            PredictorSample({"a": 1.0, "b": 2.0}, 0.1, "pde1", "m1", 1),
            PredictorSample({"a": 3.0, "b": 4.0}, 0.2, "pde1", "m2", 1),
            PredictorSample({"a": 5.0, "b": 6.0}, 0.3, "pde2", "m1", 1),
        ]
        dataset = PredictorDataset(samples=samples)
        X = dataset.feature_matrix()

        assert X.shape == (3, 2)
        assert X[0, 0] == 1.0
        assert X[2, 1] == 6.0

    def test_dataset_target_vector(self) -> None:
        from causal_pinn.training.early_predictor import (
            PredictorDataset,
            PredictorSample,
        )

        samples = [
            PredictorSample({"a": 1.0}, 0.1, "pde1", "m1", 1),
            PredictorSample({"a": 2.0}, 0.2, "pde2", "m1", 1),
        ]
        dataset = PredictorDataset(samples=samples)
        y = dataset.target_vector()

        assert y.shape == (2,)
        assert y[0] == pytest.approx(0.1)
        assert y[1] == pytest.approx(0.2)

    def test_dataset_filter_by_pde(self) -> None:
        from causal_pinn.training.early_predictor import (
            PredictorDataset,
            PredictorSample,
        )

        samples = [
            PredictorSample({"a": 1.0}, 0.1, "pde1", "m1", 1),
            PredictorSample({"a": 2.0}, 0.2, "pde2", "m1", 1),
            PredictorSample({"a": 3.0}, 0.3, "pde1", "m2", 1),
        ]
        dataset = PredictorDataset(samples=samples)
        filtered = dataset.filter_by_pde("pde1")

        assert len(filtered) == 2

    def test_dataset_unique_pdes(self) -> None:
        from causal_pinn.training.early_predictor import (
            PredictorDataset,
            PredictorSample,
        )

        samples = [
            PredictorSample({"a": 1.0}, 0.1, "pde1", "m1", 1),
            PredictorSample({"a": 2.0}, 0.2, "pde2", "m1", 1),
            PredictorSample({"a": 3.0}, 0.3, "pde1", "m2", 1),
        ]
        dataset = PredictorDataset(samples=samples)

        assert dataset.unique_pdes() == ["pde1", "pde2"]

    def test_dataset_empty(self) -> None:
        from causal_pinn.training.early_predictor import PredictorDataset

        dataset = PredictorDataset()
        assert len(dataset) == 0
        assert dataset.feature_matrix().shape == (0, 0)


class TestEarlyTrainingPredictor:
    def test_fit_and_predict(self) -> None:
        from sklearn.ensemble import RandomForestRegressor

        from causal_pinn.training.early_predictor import (
            EarlyTrainingPredictor,
            FeatureExtractor,
            PredictorDataset,
            PredictorSample,
        )

        ext = FeatureExtractor(window_size=10)
        predictor = EarlyTrainingPredictor(
            target_metric="pde_residual_norm",
            model=RandomForestRegressor(n_estimators=10, random_state=42),
            extractor=ext,
        )

        samples = []
        for i in range(20):
            losses = _make_stage2_losses(50)
            features = ext.extract(losses)
            target = 0.01 + 0.001 * i + np.random.default_rng(i).normal() * 0.001
            samples.append(
                PredictorSample(features, target, f"pde_{i % 4}", f"model_{i % 3}", i)
            )

        dataset = PredictorDataset(samples=samples)
        predictor.fit(dataset)

        assert predictor.is_fitted
        pred = predictor.predict(samples[0].features)
        assert isinstance(pred, float)
        assert pred > 0

    def test_predict_before_fit_raises(self) -> None:
        from causal_pinn.training.early_predictor import EarlyTrainingPredictor

        predictor = EarlyTrainingPredictor()
        with pytest.raises(RuntimeError, match="not been fitted"):
            predictor.predict({"total_initial": 1.0})

    def test_cross_validate(self) -> None:
        from sklearn.ensemble import RandomForestRegressor

        from causal_pinn.training.early_predictor import (
            EarlyTrainingPredictor,
            FeatureExtractor,
            PredictorDataset,
            PredictorSample,
        )

        ext = FeatureExtractor(window_size=10)
        predictor = EarlyTrainingPredictor(
            target_metric="pde_residual_norm",
            model=RandomForestRegressor(n_estimators=10, random_state=42),
            extractor=ext,
        )

        rng = np.random.default_rng(42)
        samples = []
        for pde_idx in range(3):
            base_target = 0.01 * (pde_idx + 1)
            for policy_idx in range(3):
                for seed in range(3):
                    losses = _make_stage2_losses(50)
                    features = ext.extract(losses)
                    target = base_target + 0.002 * policy_idx + rng.normal() * 0.001
                    samples.append(
                        PredictorSample(
                            features, target, f"pde_{pde_idx}", f"policy_{policy_idx}", seed
                        )
                    )

        dataset = PredictorDataset(samples=samples)
        cv_results = predictor.cross_validate(dataset, group_col="pde")

        assert "mean_r2" in cv_results
        assert "policy_selection_accuracy" in cv_results
        assert "fold_details" in cv_results
        assert len(cv_results["fold_details"]) == 3
        assert predictor.is_fitted  # refit on all data after CV

    def test_feature_importance(self) -> None:
        from sklearn.ensemble import RandomForestRegressor

        from causal_pinn.training.early_predictor import (
            EarlyTrainingPredictor,
            FeatureExtractor,
            PredictorDataset,
            PredictorSample,
        )

        ext = FeatureExtractor(window_size=10)
        predictor = EarlyTrainingPredictor(
            model=RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42),
            extractor=ext,
        )

        samples = []
        rng = np.random.default_rng(42)
        for i in range(20):
            n_epochs = 50 + i * 5
            scale = 0.5 + 0.1 * i
            losses = []
            for j in range(n_epochs):
                ep: dict = {
                    "total": scale * math.exp(-j / 50.0) + 0.01 * rng.normal(),
                    "physics_loss": scale * 0.6 * math.exp(-j / 60.0) + 0.005 * rng.normal(),
                    "bc_loss": scale * 0.2 * math.exp(-j / 40.0) + 0.002 * rng.normal(),
                    "lambda_physics": 0.1 + 0.9 * min(1.0, j / 250.0),
                    "data_loss": scale * 0.2 * math.exp(-j / 30.0) + 0.003 * rng.normal(),
                }
                losses.append(ep)
            features = ext.extract(losses)
            target = 0.005 + 0.002 * i
            samples.append(
                PredictorSample(features, target, f"pde_{i % 4}", f"m_{i % 3}", i)
            )
        dataset = PredictorDataset(samples=samples)
        predictor.fit(dataset)

        importance = predictor.feature_importance()
        assert len(importance) > 0
        assert sum(importance.values()) == pytest.approx(1.0, abs=0.05)

    def test_feature_importance_before_fit_raises(self) -> None:
        from causal_pinn.training.early_predictor import EarlyTrainingPredictor

        predictor = EarlyTrainingPredictor()
        with pytest.raises(RuntimeError, match="not been fitted"):
            predictor.feature_importance()

    def test_build_dataset_from_real_results(self, tmp_path) -> None:
        import json

        from causal_pinn.training.early_predictor import EarlyTrainingPredictor

        # Create a mock results file.
        results = {
            "experiment": "test",
            "raw_results": [
                {
                    "model_name": "hybrid_full",
                    "seed": 42,
                    "training_history": {
                        "stage2_losses": _make_stage2_losses(100),
                    },
                    "metrics": {
                        "pde_residual_norm": 0.01,
                        "relative_l2_error": 0.02,
                    },
                },
            ],
        }
        results_file = tmp_path / "results.json"
        with open(results_file, "w") as f:
            json.dump(results, f)

        predictor = EarlyTrainingPredictor(target_metric="pde_residual_norm")
        dataset = predictor.build_dataset(
            result_paths=[results_file],
            pde_labels=["test_pde"],
        )

        assert len(dataset) == 1
        assert dataset.samples[0].pde == "test_pde"
        assert dataset.samples[0].target == 0.01
        assert dataset.samples[0].policy == "hybrid_full"

    def test_build_dataset_skips_missing_target(self, tmp_path) -> None:
        import json

        from causal_pinn.training.early_predictor import EarlyTrainingPredictor

        results = {
            "raw_results": [
                {
                    "model_name": "m1",
                    "seed": 1,
                    "training_history": {"stage2_losses": _make_stage2_losses(50)},
                    "metrics": {"bc_violation": 0.001},  # no pde_residual_norm
                },
            ],
        }
        results_file = tmp_path / "results.json"
        with open(results_file, "w") as f:
            json.dump(results, f)

        predictor = EarlyTrainingPredictor(target_metric="pde_residual_norm")
        dataset = predictor.build_dataset(
            result_paths=[results_file],
            pde_labels=["test_pde"],
        )

        assert len(dataset) == 0

    def test_build_dataset_mismatched_lengths_raises(self) -> None:
        from causal_pinn.training.early_predictor import EarlyTrainingPredictor

        predictor = EarlyTrainingPredictor()
        with pytest.raises(ValueError, match="same length"):
            predictor.build_dataset(
                result_paths=["a.json"],
                pde_labels=["pde1", "pde2"],
            )

    def test_best_policy_for_target(self) -> None:
        from causal_pinn.training.early_predictor import EarlyTrainingPredictor

        policies = ["m1", "m2", "m1", "m2"]
        targets = np.array([0.1, 0.05, 0.12, 0.04])

        best = EarlyTrainingPredictor._best_policy_for_target(policies, targets)
        assert best == "m2"

    def test_predict_dataset(self) -> None:
        from sklearn.ensemble import RandomForestRegressor

        from causal_pinn.training.early_predictor import (
            EarlyTrainingPredictor,
            FeatureExtractor,
            PredictorDataset,
            PredictorSample,
        )

        ext = FeatureExtractor(window_size=10)
        predictor = EarlyTrainingPredictor(
            model=RandomForestRegressor(n_estimators=10, random_state=42),
            extractor=ext,
        )

        samples = []
        for i in range(15):
            losses = _make_stage2_losses(50)
            features = ext.extract(losses)
            target = 0.01 + 0.001 * i
            samples.append(
                PredictorSample(features, target, f"pde_{i % 3}", f"m_{i % 2}", i)
            )

        dataset = PredictorDataset(samples=samples)
        predictor.fit(dataset)
        preds = predictor.predict_dataset(dataset)

        assert preds.shape == (15,)
