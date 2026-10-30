"""PDE-aware meta-router for PINN training policy selection.

Augments early-training loss features with computable PDE characteristics
(spatial dimension, nonlinearity, derivative order, etc.) and uses a
joint MLP to predict final evaluation metrics.  The PDE features provide
structural priors that help generalize to unseen PDE types, addressing the
overfitting observed with the RF router on diverse PDE benchmarks.

Typical usage::

    from causal_pinn.training.meta_router import MetaRouter, PDE_FEATURE_REGISTRY

    router = MetaRouter()
    router.fit(training_features, pde_names, targets)
    predictions = router.predict(test_features, test_pde_names)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# PDE feature descriptors
# ---------------------------------------------------------------------------

@dataclass
class PDEFeatures:
    """Computable PDE characteristics (no training needed).

    These are static properties of the PDE that can be determined before
    any training run, providing structural priors for the router.

    Attributes:
        spatial_dim: Number of spatial dimensions (1 or 2).
        has_nonlinear_term: Whether PDE contains nonlinear terms (u*u_x, u^3, etc.).
        has_higher_order: Whether PDE has derivatives above second order (u_xxx, etc.).
        n_spatial_derivatives: Maximum spatial derivative order in the PDE.
        has_reaction_term: Whether PDE has source/reaction terms (e.g. u - u^3).
        has_exact_solution: Whether an analytical solution is available.
        domain_volume: Product of domain extents (L * t_max or L_x * L_y * t_max).
    """

    spatial_dim: int
    has_nonlinear_term: bool
    has_higher_order: bool
    n_spatial_derivatives: int
    has_reaction_term: bool
    has_exact_solution: bool
    domain_volume: float

    def to_array(self) -> NDArray[np.float64]:
        """Convert to a numeric array for model input."""
        return np.array(
            [
                self.spatial_dim,
                int(self.has_nonlinear_term),
                int(self.has_higher_order),
                self.n_spatial_derivatives,
                int(self.has_reaction_term),
                int(self.has_exact_solution),
                self.domain_volume,
            ],
            dtype=np.float64,
        )


N_PDE_FEATURES = 7

PDE_FEATURE_REGISTRY: dict[str, PDEFeatures] = {
    # --- Core PDEs ---
    # heat3d is the suite's only d=3 configuration; it keeps an analytical
    # solution (separable sine mode) so relative-L2 and therefore regret stay
    # well defined at three spatial dimensions.
    "heat3d": PDEFeatures(
        spatial_dim=3, has_nonlinear_term=False, has_higher_order=False,
        n_spatial_derivatives=2, has_reaction_term=False, has_exact_solution=True,
        domain_volume=1.0,
    ),
    # multiscale_heat2d is the suite's only configuration with separated
    # scales: two Dirichlet eigenmodes whose wavelengths differ 4x and whose
    # decay rates therefore differ 16x, so early and late dynamics are
    # dominated by different structure. The superposition stays analytic, so
    # relative-L2 and regret remain well defined.
    "multiscale_heat2d": PDEFeatures(
        spatial_dim=2, has_nonlinear_term=False, has_higher_order=False,
        n_spatial_derivatives=2, has_reaction_term=False, has_exact_solution=True,
        domain_volume=1.0,
    ),
    "heat1d": PDEFeatures(
        spatial_dim=1, has_nonlinear_term=False, has_higher_order=False,
        n_spatial_derivatives=2, has_reaction_term=False, has_exact_solution=True,
        domain_volume=1.0,
    ),
    "burgers1d": PDEFeatures(
        spatial_dim=1, has_nonlinear_term=True, has_higher_order=False,
        n_spatial_derivatives=2, has_reaction_term=False, has_exact_solution=False,
        domain_volume=1.0,
    ),
    "advdiff1d": PDEFeatures(
        spatial_dim=1, has_nonlinear_term=False, has_higher_order=False,
        n_spatial_derivatives=2, has_reaction_term=False, has_exact_solution=True,
        domain_volume=1.0,
    ),
    "kdv1d": PDEFeatures(
        spatial_dim=1, has_nonlinear_term=True, has_higher_order=True,
        n_spatial_derivatives=3, has_reaction_term=False, has_exact_solution=True,
        domain_volume=10.0,
    ),
    "allencahn1d": PDEFeatures(
        spatial_dim=1, has_nonlinear_term=True, has_higher_order=False,
        n_spatial_derivatives=2, has_reaction_term=True, has_exact_solution=True,
        domain_volume=2.0,
    ),
    "wave2d": PDEFeatures(
        spatial_dim=2, has_nonlinear_term=False, has_higher_order=False,
        n_spatial_derivatives=2, has_reaction_term=False, has_exact_solution=True,
        domain_volume=1.0,
    ),
    "navier_stokes2d": PDEFeatures(
        spatial_dim=2, has_nonlinear_term=True, has_higher_order=False,
        n_spatial_derivatives=2, has_reaction_term=False, has_exact_solution=True,
        domain_volume=39.5,
    ),
    # --- Variants (inherit structure from parent, may differ in domain) ---
    "burgers1d_lownu": PDEFeatures(
        spatial_dim=1, has_nonlinear_term=True, has_higher_order=False,
        n_spatial_derivatives=2, has_reaction_term=False, has_exact_solution=False,
        domain_volume=1.0,
    ),
    "burgers1d_verylow": PDEFeatures(
        spatial_dim=1, has_nonlinear_term=True, has_higher_order=False,
        n_spatial_derivatives=2, has_reaction_term=False, has_exact_solution=False,
        domain_volume=1.0,
    ),
    "advdiff1d_highpe": PDEFeatures(
        spatial_dim=1, has_nonlinear_term=False, has_higher_order=False,
        n_spatial_derivatives=2, has_reaction_term=False, has_exact_solution=True,
        domain_volume=0.5,
    ),
    "allencahn1d_sharp": PDEFeatures(
        spatial_dim=1, has_nonlinear_term=True, has_higher_order=False,
        n_spatial_derivatives=2, has_reaction_term=True, has_exact_solution=True,
        domain_volume=1.0,
    ),
    "reaction_diffusion1d": PDEFeatures(
        spatial_dim=1, has_nonlinear_term=True, has_higher_order=False,
        n_spatial_derivatives=2, has_reaction_term=True, has_exact_solution=False,
        domain_volume=1.0,
    ),
    "reaction_diffusion1d_stiff": PDEFeatures(
        spatial_dim=1, has_nonlinear_term=True, has_higher_order=False,
        n_spatial_derivatives=2, has_reaction_term=True, has_exact_solution=False,
        domain_volume=0.5,
    ),
}


# ---------------------------------------------------------------------------
# Meta-router
# ---------------------------------------------------------------------------

class MetaRouter:
    """PDE-aware meta-router using joint PDE + training features.

    Concatenates static PDE characteristics with early-training loss
    features and trains an MLP to predict final evaluation metrics.
    Both feature groups are independently standardized before concatenation.

    Args:
        hidden_sizes: MLP hidden layer sizes.  Defaults to (16, 8) which
            works well for the ~400-sample Paper A benchmark; larger networks
            (64, 32) overfit with leave-one-PDE-out CV.
        alpha: L2 regularization strength.  0.1 is a good default for the
            small-data regime; lower values (0.01) overfit.
        max_iter: Maximum training iterations for the MLP.
        random_state: Random seed for reproducibility.
    """

    def __init__(
        self,
        hidden_sizes: tuple[int, ...] = (16, 8),
        alpha: float = 0.1,
        max_iter: int = 2000,
        random_state: int = 42,
    ) -> None:
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_sizes,
            alpha=alpha,
            max_iter=max_iter,
            random_state=random_state,
        )
        self.scaler = StandardScaler()
        self.pde_scaler = StandardScaler()
        self.training_feature_names: list[str] = []
        self.is_fitted: bool = False

    def _pde_features_to_array(self, pde_name: str) -> NDArray[np.float64]:
        """Look up PDE features from registry, returning zeros for unknown PDEs."""
        pf = PDE_FEATURE_REGISTRY.get(pde_name)
        if pf is None:
            return np.zeros(N_PDE_FEATURES, dtype=np.float64)
        return pf.to_array()

    def _build_joint_features(
        self,
        training_features: NDArray[np.float64],
        pde_names: list[str],
        *,
        fit_scalers: bool = False,
    ) -> NDArray[np.float64]:
        """Build joint [pde_features | training_features] matrix.

        Args:
            training_features: (N, D_train) array of early-training features.
            pde_names: PDE name for each sample.
            fit_scalers: If True, fit scalers on this data (training time).

        Returns:
            (N, D_pde + D_train) standardized feature matrix.
        """
        pde_feats = np.array(
            [self._pde_features_to_array(p) for p in pde_names],
            dtype=np.float64,
        )

        if fit_scalers:
            pde_feats_scaled = self.pde_scaler.fit_transform(pde_feats)
            train_feats_scaled = self.scaler.fit_transform(training_features)
        else:
            pde_feats_scaled = self.pde_scaler.transform(pde_feats)
            train_feats_scaled = self.scaler.transform(training_features)

        return np.hstack([pde_feats_scaled, train_feats_scaled])

    def fit(
        self,
        training_features: NDArray[np.float64],
        pde_names: list[str],
        targets: NDArray[np.float64],
    ) -> MetaRouter:
        """Fit on concatenated [pde_features | training_features].

        Args:
            training_features: (N, D) array of early-training features.
            pde_names: PDE name for each of N samples.
            targets: (N,) array of final evaluation metric values.

        Returns:
            self (for method chaining).
        """
        X = self._build_joint_features(
            training_features, pde_names, fit_scalers=True,
        )
        self.model.fit(X, targets)
        self.is_fitted = True
        return self

    def predict(
        self,
        training_features: NDArray[np.float64],
        pde_names: list[str],
    ) -> NDArray[np.float64]:
        """Predict final metric for each sample.

        Args:
            training_features: (N, D) array of early-training features.
            pde_names: PDE name for each of N samples.

        Returns:
            (N,) array of predicted metric values.
        """
        if not self.is_fitted:
            raise RuntimeError("MetaRouter has not been fitted. Call fit() first.")
        X = self._build_joint_features(
            training_features, pde_names, fit_scalers=False,
        )
        return self.model.predict(X)  # type: ignore[no-any-return]
