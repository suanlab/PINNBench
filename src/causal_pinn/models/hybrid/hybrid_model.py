"""Hybrid PINN-Operator model combining neural operators with PINNs.

The HybridPINNOperator combines the strengths of neural operators (fast inference,
generalization) with PINNs (physics compliance) using a learnable causal gate
for time-aware blending.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from causal_pinn.models.hybrid.causal_gate import CausalGate

if TYPE_CHECKING:
    from causal_pinn.models.operators.base import NeuralOperator
    from causal_pinn.models.pinn.base import PINNBase
    from causal_pinn.physics.pde.base import PDEProblem


@dataclass
class HybridConfig:
    """Configuration for HybridPINNOperator.

    Attributes:
        use_operator: Whether to use neural operator predictions.
        use_pinn: Whether to use PINN predictions.
        use_causal_gate: Whether to use causal gating (if False, uses equal weighting).
        operator_weight: Weight for operator loss in total loss.
        physics_weight: Weight for physics/PINN loss in total loss.
        gate_weight: Weight for gate regularization loss (encourages balanced gating).
    """

    use_operator: bool = True
    use_pinn: bool = True
    use_causal_gate: bool = True
    operator_weight: float = 1.0
    physics_weight: float = 1.0
    gate_weight: float = 0.01


class HybridPINNOperator(nn.Module):
    """Hybrid model combining neural operator, PINN, and causal gating.

    This model combines three components:
    1. Neural Operator (e.g., FNO): Fast approximate solutions
    2. PINN: Physics-constrained solutions
    3. CausalGate: Time-aware blending with causal bias

    The model supports ablation studies via configuration flags that can
    disable individual components.

    Attributes:
        pinn_network: The PINN network for physics-constrained predictions.
        operator: The neural operator for fast approximate predictions.
        causal_gate: The gating mechanism for blending predictions.
        pde_problem: The PDE problem definition for computing physics loss.
        config: Configuration for enabling/disabling components.
    """

    def __init__(
        self,
        pinn_network: PINNBase,
        operator: NeuralOperator,
        causal_gate: CausalGate,
        pde_problem: PDEProblem,
        config: HybridConfig | None = None,
    ) -> None:
        """Initialize the HybridPINNOperator.

        Args:
            pinn_network: PINN network (e.g., PINNBase instance).
            operator: Neural operator (e.g., FNO instance).
            causal_gate: CausalGate instance for blending.
            pde_problem: PDE problem definition.
            config: Configuration for component enablement and loss weights.
        """
        super().__init__()

        self.pinn_network = pinn_network
        self.operator = operator
        self.causal_gate = causal_gate
        self.pde_problem = pde_problem
        self.config = config or HybridConfig()

    def _prepare_operator_input(self, t: Tensor, x: Tensor) -> Tensor:
        """Prepare input for the neural operator.

        Converts point-wise (t, x) coordinates to operator format.
        For FNO 1D, expects (batch, channels, spatial).

        Args:
            t: Time coordinates, shape (N,) or (N, 1).
            x: Spatial coordinates, shape (N,) or (N, 1).

        Returns:
            Operator input tensor, shape (N, 2, 1) for 1D operators.
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        combined = torch.cat([t, x], dim=-1)
        operator_dim = getattr(self.operator, "dim", 1)
        if operator_dim == 2:
            return combined.unsqueeze(-1).unsqueeze(-1)
        return combined.unsqueeze(-1)

    def _format_operator_output(self, operator_out: Tensor) -> Tensor:
        if operator_out.dim() == 4:
            operator_pred = operator_out.squeeze(-1).squeeze(-1)
        elif operator_out.dim() == 3:
            operator_pred = operator_out.squeeze(-1)
        else:
            operator_pred = operator_out

        if operator_pred.dim() == 1:
            operator_pred = operator_pred.unsqueeze(-1)

        return operator_pred

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        """Compute blended prediction from operator and PINN.

        Args:
            t: Time coordinates, shape (N,) or (N, 1).
            x: Spatial coordinates, shape (N,) or (N, 1).

        Returns:
            Blended prediction tensor, shape (N, out_dim).
        """
        # Ensure 2D tensors
        t_2d = t.unsqueeze(-1) if t.dim() == 1 else t
        x_2d = x.unsqueeze(-1) if x.dim() == 1 else x

        # Get PINN prediction
        if self.config.use_pinn:
            pinn_input = torch.cat([t_2d, x_2d], dim=-1)
            pinn_pred = self.pinn_network(pinn_input)
        else:
            pinn_pred = torch.zeros(t_2d.shape[0], self.causal_gate.out_dim, device=t.device)

        # Get operator prediction
        if self.config.use_operator:
            operator_input = self._prepare_operator_input(t_2d, x_2d)
            operator_out = self.operator(operator_input)
            operator_pred = self._format_operator_output(operator_out)
        else:
            operator_pred = torch.zeros(t_2d.shape[0], self.causal_gate.out_dim, device=t.device)

        # Blend predictions
        if self.config.use_causal_gate and self.config.use_operator and self.config.use_pinn:
            blended_pred, _ = self.causal_gate(operator_pred, pinn_pred, t_2d)
        elif self.config.use_operator and not self.config.use_pinn:
            blended_pred = operator_pred
        elif self.config.use_pinn and not self.config.use_operator:
            blended_pred = pinn_pred
        else:
            # Neither enabled - return zeros (should not happen in practice)
            blended_pred = torch.zeros_like(pinn_pred)

        return blended_pred  # type: ignore[no-any-return]

    def compute_hybrid_loss(
        self,
        t: Tensor,
        x: Tensor,
    ) -> dict[str, Tensor | dict[str, Tensor]]:
        """Compute the hybrid loss with all components.

        Computes losses from:
        1. Operator prediction (data/supervised loss)
        2. Physics loss (PDE residual from PINN)
        3. Causal loss (optional, for causal training)
        4. Gate statistics for monitoring

        Args:
            t: Time coordinates, shape (N,) or (N, 1). Must have requires_grad=True
               for physics loss computation.
            x: Spatial coordinates, shape (N,) or (N, 1). Must have requires_grad=True
               for physics loss computation.

        Returns:
            Dictionary containing:
                - operator_loss: Loss from operator predictions
                - physics_loss: PDE residual loss
                - causal_loss: Causal-weighted loss (if applicable)
                - gate_stats: Dictionary of gate value statistics
                - total: Weighted total loss
        """
        # Ensure gradients are tracked for physics loss
        # Important: Don't clone if already requires_grad, to preserve computational graph
        if not t.requires_grad:
            t = t.clone().requires_grad_(True)
        if not x.requires_grad:
            x = x.clone().requires_grad_(True)

        # Ensure 2D tensors (use views to preserve computational graph)
        t_2d = t if t.dim() == 2 else t.unsqueeze(-1)
        x_2d = x if x.dim() == 2 else x.unsqueeze(-1)

        losses: dict[str, Tensor] = {}
        gate_stats: dict[str, Tensor] = {}

        # Get PINN prediction and compute physics loss
        if self.config.use_pinn:
            pinn_input = torch.cat([t_2d, x_2d], dim=-1)
            pinn_pred = self.pinn_network(pinn_input)

            # Compute PDE residual for physics loss
            # Pass original t and x tensors to preserve computational graph for autograd
            # Squeeze u to 1D to match expected input shape for PDE residual method
            u = pinn_pred.squeeze(-1) if pinn_pred.dim() > 1 else pinn_pred
            residual = self.pde_problem.residual(u, t, x)
            physics_loss = torch.mean(residual**2)
            losses["physics_loss"] = physics_loss
        else:
            pinn_pred = torch.zeros(t_2d.shape[0], self.causal_gate.out_dim, device=t.device)
            losses["physics_loss"] = torch.tensor(0.0, device=t.device)
        # Get operator prediction
        if self.config.use_operator:
            operator_input = self._prepare_operator_input(t_2d, x_2d)
            operator_out = self.operator(operator_input)
            operator_pred = self._format_operator_output(operator_out)

            # Operator self-consistency loss (prediction should be smooth)
            # Using gradient-based regularization
            if operator_pred.requires_grad:
                op_grad = torch.autograd.grad(
                    operator_pred.sum(),
                    operator_input,
                    create_graph=True,
                    retain_graph=True,
                )[0]
                operator_loss = torch.mean(op_grad**2) * 0.01  # Small regularization
            else:
                operator_loss = torch.tensor(0.0, device=t.device)
            losses["operator_loss"] = operator_loss
        else:
            operator_pred = torch.zeros(t_2d.shape[0], self.causal_gate.out_dim, device=t.device)
            losses["operator_loss"] = torch.tensor(0.0, device=t.device)

        # Compute gate and blending
        if self.config.use_causal_gate and self.config.use_operator and self.config.use_pinn:
            blended_pred, gate_values = self.causal_gate(operator_pred, pinn_pred, t_2d)
            gate_stats = self.causal_gate.get_gate_stats(gate_values)

            # Gate regularization: encourage balanced use of both components
            # Penalize gate values too close to 0 or 1
            gate_entropy = -(
                gate_values * torch.log(gate_values + 1e-8)
                + (1 - gate_values) * torch.log(1 - gate_values + 1e-8)
            )
            gate_reg_loss = -torch.mean(gate_entropy)
            losses["gate_reg_loss"] = gate_reg_loss

            # Causal loss: use blended prediction for residual
            # Pass original t and x tensors to preserve computational graph
            # Squeeze blended_pred to 1D to match expected input shape for PDE residual
            u_blended = blended_pred.squeeze(-1) if blended_pred.dim() > 1 else blended_pred
            residual_blended = self.pde_problem.residual(u_blended, t, x)
            causal_loss = torch.mean(residual_blended**2)
            losses["causal_loss"] = causal_loss
        else:
            gate_stats = {
                "mean": torch.tensor(0.5, device=t.device),
                "std": torch.tensor(0.0, device=t.device),
                "min": torch.tensor(0.5, device=t.device),
                "max": torch.tensor(0.5, device=t.device),
                "frac_operator": torch.tensor(0.5, device=t.device),
                "frac_pinn": torch.tensor(0.5, device=t.device),
            }
            losses["gate_reg_loss"] = torch.tensor(0.0, device=t.device)
            losses["causal_loss"] = torch.tensor(0.0, device=t.device)

        # Compute total weighted loss
        total_loss = (
            self.config.operator_weight * losses["operator_loss"]
            + self.config.physics_weight * losses["physics_loss"]
            + self.config.gate_weight * losses["gate_reg_loss"]
        )

        return {
            "operator_loss": losses["operator_loss"],
            "physics_loss": losses["physics_loss"],
            "causal_loss": losses["causal_loss"],
            "gate_reg_loss": losses["gate_reg_loss"],
            "gate_stats": gate_stats,
            "total": total_loss,
        }

    def get_component_predictions(
        self,
        t: Tensor,
        x: Tensor,
    ) -> dict[str, Tensor]:
        """Get individual predictions from each component for analysis.

        Args:
            t: Time coordinates, shape (N,) or (N, 1).
            x: Spatial coordinates, shape (N,) or (N, 1).

        Returns:
            Dictionary containing:
                - pinn_pred: PINN prediction (N, out_dim)
                - operator_pred: Operator prediction (N, out_dim)
                - blended_pred: Blended prediction (N, out_dim)
                - gate_values: Gate values (N, 1) or None
        """
        # Ensure 2D tensors
        t_2d = t.unsqueeze(-1) if t.dim() == 1 else t
        x_2d = x.unsqueeze(-1) if x.dim() == 1 else x

        result: dict[str, Tensor | None] = {}

        # PINN prediction
        if self.config.use_pinn:
            pinn_input = torch.cat([t_2d, x_2d], dim=-1)
            result["pinn_pred"] = self.pinn_network(pinn_input)
        else:
            result["pinn_pred"] = None

        # Operator prediction
        if self.config.use_operator:
            operator_input = self._prepare_operator_input(t_2d, x_2d)
            operator_out = self.operator(operator_input)
            operator_pred = self._format_operator_output(operator_out)
            result["operator_pred"] = operator_pred
        else:
            result["operator_pred"] = None

        # Blended prediction and gate values
        if (
            self.config.use_causal_gate
            and result["pinn_pred"] is not None
            and result["operator_pred"] is not None
        ):
            blended_pred, gate_values = self.causal_gate(
                result["operator_pred"], result["pinn_pred"], t_2d
            )
            result["blended_pred"] = blended_pred
            result["gate_values"] = gate_values
        elif result["pinn_pred"] is not None:
            result["blended_pred"] = result["pinn_pred"]
            result["gate_values"] = None
        elif result["operator_pred"] is not None:
            result["blended_pred"] = result["operator_pred"]
            result["gate_values"] = None
        else:
            result["blended_pred"] = None
            result["gate_values"] = None

        return result  # type: ignore

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"HybridPINNOperator("
            f"use_operator={self.config.use_operator}, "
            f"use_pinn={self.config.use_pinn}, "
            f"use_causal_gate={self.config.use_causal_gate})"
        )



class HybridModelWrapper(nn.Module):
    """Wraps HybridPINNOperator for compatibility with HybridTrainer.

    HybridTrainer calls model(inputs) where inputs is (N, 2) [t, x concatenated].
    HybridPINNOperator.forward(t, x) expects separate tensors.
    This wrapper bridges the interface gap.

    Also handles the edge case where use_causal_gate=False with both operator
    and PINN enabled (HybridPINNOperator.forward returns zeros in this case).
    """

    def __init__(self, hybrid_model: HybridPINNOperator) -> None:
        super().__init__()
        self.hybrid_model = hybrid_model

    def forward(self, inputs: Tensor) -> Tensor:
        t = inputs[:, 0:1]
        x = inputs[:, 1:] if inputs.shape[1] > 2 else inputs[:, 1:2]

        # Check if we need to handle the no-gate case manually
        config = self.hybrid_model.config
        if config.use_operator and config.use_pinn and not config.use_causal_gate:
            # HybridPINNOperator.forward returns zeros for this config,
            # so we manually compute and blend predictions
            t_2d = t
            x_2d = x

            # Get PINN prediction
            pinn_input = torch.cat([t_2d, x_2d], dim=-1)
            pinn_pred = self.hybrid_model.pinn_network(pinn_input)

            # Get operator prediction
            operator_input = self.hybrid_model._prepare_operator_input(t_2d, x_2d)
            operator_out = self.hybrid_model.operator(operator_input)
            operator_pred = self.hybrid_model._format_operator_output(operator_out)

            # Simple 50/50 blend for no-gate case
            return 0.5 * pinn_pred + 0.5 * operator_pred  # type: ignore[no-any-return]
        else:
            # Normal case: delegate to hybrid model
            return self.hybrid_model(t, x)  # type: ignore[no-any-return]
