"""Physics losses module for PINN training."""

from causal_pinn.physics.losses.adaptive_epsilon import AdaptiveEpsilonScheduler
from causal_pinn.physics.losses.boundary_loss import (
    BoundaryLoss,
    DirichletBC,
    NeumannBC,
)
from causal_pinn.physics.losses.causal_loss import (
    AdaptiveCausalLoss,
    CausalLoss,
    TimeDecayLoss,
)
from causal_pinn.physics.losses.physics_loss import (
    HigherOrderPhysicsLoss,
    PhysicsLoss,
)

__all__ = [
    # Physics losses
    "PhysicsLoss",
    "HigherOrderPhysicsLoss",
    # Boundary losses
    "BoundaryLoss",
    "DirichletBC",
    "NeumannBC",
    # Causal losses
    "CausalLoss",
    "AdaptiveCausalLoss",
    "TimeDecayLoss",
    # Adaptive epsilon
    "AdaptiveEpsilonScheduler",
]
