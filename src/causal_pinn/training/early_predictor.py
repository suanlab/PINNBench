"""Early-training predictor for PINN training policy selection.

Given the first N epochs of standard stage-2 loss signals from a PINN training
run, predicts the final evaluation metric (pde_residual_norm or relative_l2_error)
and recommends the best training policy for a given PDE.

The predictor uses summary statistics extracted from loss curves (not raw time
series) as input features, making it lightweight and interpretable.  The default
model is a scikit-learn RandomForestRegressor, but any regressor with a
``fit(X, y)`` / ``predict(X)`` interface can be substituted.

Typical usage::

    extractor = FeatureExtractor(window_size=50)
    predictor = EarlyTrainingPredictor(target_metric="pde_residual_norm")

    # Collect training data from existing experiment results
    dataset = predictor.build_dataset(
        result_paths=[
            "experiments/results/rq3_10seed/results.json",
            "experiments/results/rq4_10seed/results.json",
        ],
        pde_labels=["heat1d", "heat1d", "heat1d", "wave2d", "wave2d"],
        extractor=extractor,
    )

    # Train with leave-one-PDE-out cross-validation
    cv_results = predictor.cross_validate(dataset, group_col="pde")
    print(cv_results)

    # Train final model on all data
    predictor.fit(dataset)

    # Predict for a new run (first 50 epochs of stage-2 losses)
    features = extractor.extract(stage2_losses[:50])
    predicted_metric = predictor.predict(features)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

# Keys that are always available in stage-2 loss dicts (when present).
_LOSS_KEYS = ["total", "physics_loss", "bc_loss"]
_OPTIONAL_LOSS_KEYS = ["data_loss", "lambda_physics"]


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_std(values: list[float]) -> float:
    return float(np.std(values)) if len(values) > 1 else 0.0


def _safe_min(values: list[float]) -> float:
    return float(np.min(values)) if values else 0.0


def _linear_slope(values: list[float]) -> float:
    """Ordinary-least-squares slope of *values* indexed 0..len-1."""
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    y = np.array(values, dtype=float)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom < 1e-12:
        return 0.0
    return float(np.sum((x - x_mean) * (y - y_mean)) / denom)


@dataclass
class FeatureExtractor:
    """Extract tabular features from the first *window_size* epochs of
    stage-2 loss history.

    For each available loss component (total, physics_loss, bc_loss,
    data_loss, lambda_physics) the following summary statistics are
    computed over the observation window:

    - ``{key}_initial``  — value at epoch 0
    - ``{key}_final``    — value at last epoch in window
    - ``{key}_mean``     — mean over window
    - ``{key}_std``      — standard deviation over window
    - ``{key}_min``      — minimum value
    - ``{key}_slope``    — linear trend (OLS slope)
    - ``{key}_ratio``    — final / max(initial, eps)

    Additionally, ratio features are computed between components:
    - ``physics_data_ratio_final``  — physics_loss / data_loss at last epoch
    - ``bc_total_ratio_final``      — bc_loss / total at last epoch

    Attributes:
        window_size: Number of early-stage epochs to observe.
        loss_keys: Loss component keys to extract features for.
        ratio_pairs: Pairs of (numerator_key, denominator_key) for ratio
            features at the final observed epoch.
    """

    window_size: int = 50
    loss_keys: list[str] = field(
        default_factory=lambda: _LOSS_KEYS + _OPTIONAL_LOSS_KEYS
    )
    ratio_pairs: list[tuple[str, str]] = field(
        default_factory=lambda: [
            ("physics_loss", "data_loss"),
            ("bc_loss", "total"),
            ("physics_loss", "total"),
        ]
    )
    include_curvature: bool = False
    curvature_keys: list[str] = field(
        default_factory=lambda: ["total", "physics_loss"]
    )

    def extract(self, stage2_losses: list[dict[str, float]]) -> dict[str, float]:
        """Extract features from a list of per-epoch loss dicts.

        Args:
            stage2_losses: List of dicts, one per epoch, each containing
                loss component values (e.g. ``{"total": 0.5, ...}``).

        Returns:
            Dict mapping feature name → float value.
        """
        window = stage2_losses[: self.window_size]
        if not window:
            return {}

        features: dict[str, float] = {}

        for key in self.loss_keys:
            values = [ep.get(key, float("nan")) for ep in window]
            # Skip keys that are entirely NaN (e.g. data_loss missing).
            if all(math.isnan(v) for v in values):
                continue
            # Replace NaN with 0.0 for computation.
            clean = [0.0 if math.isnan(v) else v for v in values]

            initial = clean[0]
            final = clean[-1]
            eps = max(abs(initial), 1e-8)

            features[f"{key}_initial"] = initial
            features[f"{key}_final"] = final
            features[f"{key}_mean"] = _safe_mean(clean)
            features[f"{key}_std"] = _safe_std(clean)
            features[f"{key}_min"] = _safe_min(clean)
            features[f"{key}_slope"] = _linear_slope(clean)
            features[f"{key}_ratio"] = final / eps

        # Ratio features at final epoch.
        final_ep = window[-1]
        for num_key, den_key in self.ratio_pairs:
            num_val = final_ep.get(num_key, float("nan"))
            den_val = final_ep.get(den_key, float("nan"))
            if math.isnan(num_val) or math.isnan(den_val) or abs(den_val) < 1e-12:
                continue
            features[f"{num_key}_{den_key}_ratio_final"] = num_val / den_val

        # Curvature (second-difference) features.
        if self.include_curvature:
            for key in self.curvature_keys:
                values = [ep.get(key, float("nan")) for ep in window]
                if all(math.isnan(v) for v in values):
                    continue
                clean = [0.0 if math.isnan(v) else v for v in values]
                if len(clean) >= 3:
                    second_diffs = [
                        clean[i + 2] - 2.0 * clean[i + 1] + clean[i]
                        for i in range(len(clean) - 2)
                    ]
                    features[f"{key}_curvature_mean"] = _safe_mean(second_diffs)
                    features[f"{key}_curvature_final"] = second_diffs[-1]

        return features

    @property
    def feature_names(self) -> list[str]:
        """Return canonical feature names in extraction order.

        Note: actual features returned by ``extract()`` may differ if
        some loss keys are missing from the input data.
        """
        names: list[str] = []
        for key in self.loss_keys:
            for suffix in [
                "initial",
                "final",
                "mean",
                "std",
                "min",
                "slope",
                "ratio",
            ]:
                names.append(f"{key}_{suffix}")
        for num_key, den_key in self.ratio_pairs:
            names.append(f"{num_key}_{den_key}_ratio_final")
        if self.include_curvature:
            for key in self.curvature_keys:
                names.append(f"{key}_curvature_mean")
                names.append(f"{key}_curvature_final")
        return names

    @property
    def n_features(self) -> int:
        return len(self.feature_names)


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

@dataclass
class PredictorSample:
    """A single (features, target, metadata) sample for the predictor.

    Attributes:
        features: Dict of feature name → value.
        target: The predicted evaluation metric value.
        pde: PDE identifier string (e.g. "heat1d").
        policy: Training policy / model name (e.g. "hybrid_full").
        seed: Random seed of the run.
    """

    features: dict[str, float]
    target: float
    pde: str
    policy: str
    seed: int


@dataclass
class PredictorDataset:
    """Collection of predictor samples with array accessors.

    Attributes:
        samples: List of PredictorSample instances.
    """

    samples: list[PredictorSample] = field(default_factory=list)

    def feature_matrix(
        self, feature_names: list[str] | None = None,
    ) -> NDArray[np.float64]:
        """Build (N, D) feature matrix.

        If *feature_names* is provided, uses exactly those columns (missing
        keys default to 0.0).  Otherwise, uses the union of all keys
        observed across samples.
        """
        if not self.samples:
            return np.array([], dtype=np.float64).reshape(0, 0)
        if feature_names is None:
            feature_names = []
            seen: set[str] = set()
            for s in self.samples:
                for k in s.features:
                    if k not in seen:
                        feature_names.append(k)
                        seen.add(k)
        rows = []
        for s in self.samples:
            rows.append([s.features.get(k, 0.0) for k in feature_names])
        return np.array(rows, dtype=np.float64)

    def target_vector(self) -> NDArray[np.float64]:
        return np.array([s.target for s in self.samples], dtype=np.float64)

    def pde_labels(self) -> list[str]:
        return [s.pde for s in self.samples]

    def policy_labels(self) -> list[str]:
        return [s.policy for s in self.samples]

    def filter_by_pde(self, pde: str) -> PredictorDataset:
        return PredictorDataset(samples=[s for s in self.samples if s.pde == pde])

    def unique_pdes(self) -> list[str]:
        seen: list[str] = []
        for s in self.samples:
            if s.pde not in seen:
                seen.append(s.pde)
        return seen

    def __len__(self) -> int:
        return len(self.samples)


# ---------------------------------------------------------------------------
# Result loading
# ---------------------------------------------------------------------------

def load_experiment_results(
    path: str | Path,
) -> dict[str, Any]:
    """Load an experiment results JSON file."""
    with open(path) as f:
        return json.load(f)  # type: ignore[no-any-return]


def extract_runs(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract individual runs from experiment results dict.

    Handles both ``raw_results`` and ``runs`` keys.
    """
    return data.get("raw_results") or data.get("runs") or []


def get_final_metrics(run: dict[str, Any]) -> dict[str, Any]:
    """Get final evaluation metrics from a run dict."""
    return run.get("metrics") or run.get("evaluation_metrics") or {}


def get_training_history(run: dict[str, Any]) -> dict[str, Any]:
    """Get training history from a run dict."""
    return run.get("training_history") or {}


# ---------------------------------------------------------------------------
# Early-training predictor
# ---------------------------------------------------------------------------

class EarlyTrainingPredictor:
    """Predict final PINN evaluation metric from early-stage loss signals.

    Uses a scikit-learn RandomForestRegressor by default.  Supports
    leave-one-PDE-out cross-validation for evaluating generalization
    to unseen PDEs.

    Attributes:
        target_metric: Name of the evaluation metric to predict
            (e.g. ``"pde_residual_norm"`` or ``"relative_l2_error"``).
        model: The underlying scikit-learn regressor.
        extractor: FeatureExtractor instance.
        feature_names_: Feature names seen during the last ``fit()`` call.
        is_fitted: Whether the model has been fitted.
    """

    def __init__(
        self,
        target_metric: str = "pde_residual_norm",
        model: Any | None = None,
        extractor: FeatureExtractor | None = None,
    ) -> None:
        self.target_metric = target_metric
        self.extractor = extractor or FeatureExtractor(window_size=50)
        self.feature_names_: list[str] = []
        self.is_fitted = False

        if model is not None:
            self.model = model
        else:
            from sklearn.ensemble import RandomForestRegressor

            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1,
            )

    # ------------------------------------------------------------------
    # Dataset construction
    # ------------------------------------------------------------------

    def build_dataset(
        self,
        result_paths: list[str | Path],
        pde_labels: list[str],
        extractor: FeatureExtractor | None = None,
    ) -> PredictorDataset:
        """Build a PredictorDataset from experiment result files.

        Args:
            result_paths: Paths to experiment results JSON files.
            pde_labels: PDE label for each result file (must match length).
            extractor: FeatureExtractor to use (defaults to self.extractor).

        Returns:
            PredictorDataset with one sample per (PDE, policy, seed) run.

        Raises:
            ValueError: If lengths of *result_paths* and *pde_labels* differ.
        """
        if len(result_paths) != len(pde_labels):
            raise ValueError(
                f"result_paths ({len(result_paths)}) and pde_labels "
                f"({len(pde_labels)}) must have the same length",
            )

        ext = extractor or self.extractor
        dataset = PredictorDataset()

        for path, pde in zip(result_paths, pde_labels, strict=True):
            data = load_experiment_results(path)
            runs = extract_runs(data)

            for run in runs:
                history = get_training_history(run)
                stage2 = history.get("stage2_losses", [])

                if not stage2:
                    continue

                metrics = get_final_metrics(run)
                target = metrics.get(self.target_metric)
                if target is None:
                    continue

                policy = run.get("model_name") or run.get("condition") or "unknown"
                seed = run.get("seed", 0)

                features = ext.extract(stage2)
                if not features:
                    continue

                dataset.samples.append(
                    PredictorSample(
                        features=features,
                        target=float(target),
                        pde=pde,
                        policy=policy,
                        seed=seed,
                    )
                )

        return dataset

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, dataset: PredictorDataset) -> EarlyTrainingPredictor:
        """Fit the predictor model on a dataset.

        Args:
            dataset: PredictorDataset with samples.

        Returns:
            self (for method chaining).
        """
        X = dataset.feature_matrix()
        y = dataset.target_vector()
        self.model.fit(X, y)
        self.is_fitted = True
        self.feature_names_ = list(dataset.samples[0].features.keys()) if dataset.samples else []
        return self

    def predict(self, features: dict[str, float]) -> float:
        """Predict the final metric for a single run.

        Args:
            features: Dict of feature name → value (from FeatureExtractor).

        Returns:
            Predicted evaluation metric value.
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        # Align to feature order seen during training.
        row = [features.get(k, 0.0) for k in self.feature_names_]
        X_single = np.array([row], dtype=np.float64)
        return float(self.model.predict(X_single)[0])

    def predict_dataset(self, dataset: PredictorDataset) -> NDArray[np.float64]:
        """Predict targets for an entire dataset.

        Args:
            dataset: PredictorDataset with samples.

        Returns:
            1-D array of predicted metric values.
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        X_pred = dataset.feature_matrix()
        # Reorder columns to match training order.
        if self.feature_names_:
            all_names = list(dataset.samples[0].features.keys()) if dataset.samples else []
            col_order = [all_names.index(k) if k in all_names else -1 for k in self.feature_names_]
            valid = [i for i in col_order if i >= 0]
            if valid:
                X_pred = X_pred[:, valid]
        return self.model.predict(X_pred)  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def cross_validate(
        self,
        dataset: PredictorDataset,
        group_col: str = "pde",
    ) -> dict[str, Any]:
        """Leave-one-group-out cross-validation.

        Args:
            dataset: Full dataset.
            group_col: Metadata field to group by (``"pde"`` for
                leave-one-PDE-out).

        Returns:
            Dict with keys:
                - ``r2_scores``: Per-fold R² scores.
                - ``mean_r2``: Mean R² across folds.
                - ``policy_selection_accuracy``: Fraction of folds where
                    the predicted best policy matches the actual best policy.
                - ``fold_details``: Per-fold breakdown.
        """
        groups = dataset.pde_labels() if group_col == "pde" else [
            getattr(s, group_col, "unknown") for s in dataset.samples
        ]
        unique_groups = list(dict.fromkeys(groups))

        # Compute global feature set so train/test matrices have same columns.
        global_names: list[str] = []
        seen_names: set[str] = set()
        for s in dataset.samples:
            for k in s.features:
                if k not in seen_names:
                    global_names.append(k)
                    seen_names.add(k)

        r2_scores: list[float] = []
        fold_details: list[dict[str, Any]] = []
        policy_correct = 0
        total_folds = 0

        for held_out_pde in unique_groups:
            train_mask = [g != held_out_pde for g in groups]
            test_mask = [g == held_out_pde for g in groups]
            train_ds = PredictorDataset(
                samples=[s for s, m in zip(dataset.samples, train_mask, strict=True) if m]
            )
            test_ds = PredictorDataset(
                samples=[s for s, m in zip(dataset.samples, test_mask, strict=True) if m]
            )

            if train_ds.__len__() == 0 or test_ds.__len__() == 0:
                continue

            # Fit on train.
            X_train = train_ds.feature_matrix(feature_names=global_names)
            y_train = train_ds.target_vector()
            self.model.fit(X_train, y_train)

            # Predict on test.
            X_test = test_ds.feature_matrix(feature_names=global_names)
            y_test = test_ds.target_vector()
            y_pred = self.model.predict(X_test)

            # R² score.
            ss_res = np.sum((y_test - y_pred) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
            r2_scores.append(r2)

            # Policy selection: predicted best vs actual best.
            test_policies = test_ds.policy_labels()
            actual_best_policy = self._best_policy_for_target(
                test_policies, y_test
            )
            predicted_best_policy = self._best_policy_for_target(
                test_policies, y_pred
            )
            is_correct = actual_best_policy == predicted_best_policy
            policy_correct += int(is_correct)
            total_folds += 1

            fold_details.append({
                "held_out_pde": held_out_pde,
                "n_train": train_ds.__len__(),
                "n_test": test_ds.__len__(),
                "r2": r2,
                "actual_best_policy": actual_best_policy,
                "predicted_best_policy": predicted_best_policy,
                "policy_correct": is_correct,
                "mean_actual_target": float(np.mean(y_test)),
                "mean_predicted_target": float(np.mean(y_pred)),
            })

        mean_r2 = float(np.mean(r2_scores)) if r2_scores else 0.0
        policy_acc = policy_correct / total_folds if total_folds > 0 else 0.0

        # Restore model state (refit on all data).
        self.fit(dataset)

        return {
            "r2_scores": r2_scores,
            "mean_r2": mean_r2,
            "policy_selection_accuracy": policy_acc,
            "fold_details": fold_details,
        }

    @staticmethod
    def _best_policy_for_target(
        policies: list[str], targets: NDArray[np.float64]
    ) -> str:
        """Return the policy with the lowest (best) target value."""
        if not policies:
            return "unknown"
        # Group by policy, take mean target.
        policy_targets: dict[str, list[float]] = {}
        for p, t in zip(policies, targets, strict=True):
            policy_targets.setdefault(p, []).append(float(t))
        policy_means = {p: float(np.mean(ts)) for p, ts in policy_targets.items()}
        best_policy = min(policy_means, key=lambda k: policy_means[k])
        return best_policy

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importance(self) -> dict[str, float]:
        """Return feature importance scores from the fitted model.

        Returns:
            Dict mapping feature name → importance score.
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        if not hasattr(self.model, "feature_importances_"):
            return {}

        importances = self.model.feature_importances_
        return dict(zip(self.feature_names_, importances.tolist(), strict=True))
