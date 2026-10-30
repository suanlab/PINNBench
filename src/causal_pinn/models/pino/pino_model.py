from __future__ import annotations

import torch
from torch import Tensor, nn

from ...physics.pde.base import PDEProblem
from ..operators.fno import FNO


class PINOModel(nn.Module):
    def __init__(self, fno: FNO, pde_problem: PDEProblem) -> None:
        super().__init__()
        self.fno = fno
        self.pde_problem = pde_problem

    def _prepare_operator_input(self, t: Tensor, x: Tensor) -> Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        if t.shape[1] != 1 or x.shape[1] != 1:
            raise ValueError("PINOModel supports only 1D space input with shapes (N, 1).")

        return torch.cat([t, x], dim=-1).unsqueeze(-1)

    def _flatten_operator_output(self, operator_out: Tensor) -> Tensor:
        if operator_out.dim() == 3:
            operator_out = operator_out.squeeze(-1)
        if operator_out.dim() == 1:
            operator_out = operator_out.unsqueeze(-1)
        return operator_out

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.dim() != 2 or inputs.shape[1] != 2:
            raise ValueError("PINOModel forward expects inputs with shape (N, 2).")

        t = inputs[:, 0:1]
        x = inputs[:, 1:2]
        operator_input = self._prepare_operator_input(t, x)
        operator_out = self.fno(operator_input)
        return self._flatten_operator_output(operator_out)

    def compute_pino_loss(self, t_colloc: Tensor, x_colloc: Tensor) -> dict[str, Tensor]:
        if not t_colloc.requires_grad:
            t_colloc = t_colloc.clone().requires_grad_(True)
        if not x_colloc.requires_grad:
            x_colloc = x_colloc.clone().requires_grad_(True)

        t_for_model = t_colloc.unsqueeze(-1) if t_colloc.dim() == 1 else t_colloc
        x_for_model = x_colloc.unsqueeze(-1) if x_colloc.dim() == 1 else x_colloc
        model_inputs = torch.cat([t_for_model, x_for_model], dim=-1)

        u_pred = self.forward(model_inputs)
        residual = self.pde_problem.residual(u_pred.squeeze(-1), t_colloc, x_colloc)
        physics_loss = torch.mean(residual**2)

        bc_target = self.pde_problem.boundary_condition(t_colloc, x_colloc)
        bc_loss = torch.mean((u_pred.squeeze(-1) - bc_target.reshape(-1)) ** 2)

        data_loss = torch.zeros((), dtype=u_pred.dtype, device=u_pred.device)
        try:
            u_exact = self.pde_problem.exact_solution(t_colloc.detach(), x_colloc.detach())
        except (NotImplementedError, AttributeError, TypeError):
            u_exact = None
        if u_exact is not None:
            data_loss = torch.mean((u_pred.squeeze(-1) - u_exact.reshape(-1)) ** 2)

        total = physics_loss + bc_loss + data_loss
        return {
            "physics_loss": physics_loss,
            "data_loss": data_loss,
            "bc_loss": bc_loss,
            "total": total,
        }
