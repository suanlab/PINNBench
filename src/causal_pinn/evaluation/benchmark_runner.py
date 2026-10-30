"""Benchmark automation runner for PINN model evaluation.

This module provides tools for running comprehensive benchmarks
comparing different PINN models and configurations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor, nn

from .metrics import (
    bc_violation,
    ic_violation,
    mae,
    max_error,
    mse,
    pde_residual_norm,
    relative_l2_error,
)
from .physics_consistency import PhysicsConsistencyChecker

if TYPE_CHECKING:
    from causal_pinn.physics.pde.base import PDEProblem


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs.

    Attributes:
        n_collocation: Number of collocation points for residual evaluation.
        n_boundary: Number of boundary points for BC evaluation.
        n_initial: Number of initial condition points.
        include_physics_consistency: Whether to compute physics consistency metrics.
        device: Torch device for computations.
        metric_names: List of metrics to compute.
    """

    n_collocation: int = 1000
    n_boundary: int = 100
    n_initial: int = 100
    include_physics_consistency: bool = True
    device: str = 'cpu'
    metric_names: list[str] = field(
        default_factory=lambda: [
            'relative_l2_error',
            'mse',
            'mae',
            'max_error',
            'pde_residual_norm',
            'bc_violation',
            'ic_violation',
        ]
    )


class BenchmarkRunner:
    """Runner for automated PINN benchmarks.

    This class provides methods to run benchmarks on single models
    and compare multiple models against each other.

    Attributes:
        config: Benchmark configuration.
        physics_checker: PhysicsConsistencyChecker instance.
    """

    def __init__(self, config: BenchmarkConfig | None = None) -> None:
        """Initialize the benchmark runner.

        Args:
            config: Benchmark configuration. Uses defaults if not provided.
        """
        self.config = config or BenchmarkConfig()
        self.physics_checker = PhysicsConsistencyChecker(device=self.config.device)

    def _generate_collocation_points(
        self,
        pde_problem: PDEProblem
    ) -> tuple[Tensor, Tensor]:
        """Generate random collocation points in the domain.

        Args:
            pde_problem: The PDE problem defining the domain.

        Returns:
            Tuple of (t, x) coordinate tensors.
        """
        t_bounds, x_bounds_raw = pde_problem.domain_bounds
        x_bounds = cast(tuple[float, ...], x_bounds_raw)
        t_min, t_max = t_bounds

        t = (
            torch.rand(self.config.n_collocation, device=self.config.device) * (t_max - t_min)
            + t_min
        )
        x = self._sample_spatial_points(self.config.n_collocation, x_bounds)

        return t, x

    def _generate_boundary_points(
        self,
        pde_problem: PDEProblem
    ) -> tuple[Tensor, Tensor]:
        """Generate points at domain boundaries.

        Args:
            pde_problem: The PDE problem defining the domain.

        Returns:
            Tuple of (t, x) coordinate tensors at boundaries.
        """
        t_bounds, x_bounds_raw = pde_problem.domain_bounds
        x_bounds = cast(tuple[float, ...], x_bounds_raw)
        t_min, t_max = t_bounds
        spatial_dim = self._infer_spatial_dim(x_bounds)

        if spatial_dim == 1:
            x_min, x_max = x_bounds[0], x_bounds[1]
            n_half = self.config.n_boundary // 2
            n_rest = self.config.n_boundary - n_half

            t_left = (
                torch.rand(n_half, device=self.config.device) * (t_max - t_min)
                + t_min
            )
            x_left = torch.full((n_half,), x_min, device=self.config.device)

            t_right = (
                torch.rand(n_rest, device=self.config.device) * (t_max - t_min)
                + t_min
            )
            x_right = torch.full((n_rest,), x_max, device=self.config.device)

            return torch.cat([t_left, t_right]), torch.cat([x_left, x_right])

        if spatial_dim >= 3:
            # Generic box boundary: 2*d faces, each with one coordinate pinned to
            # a bound and the remaining ones sampled uniformly. The 1D and 2D
            # paths above are kept as they were so their sampling is unchanged.
            n_faces = 2 * spatial_dim
            per_face = [self.config.n_boundary // n_faces] * n_faces
            for idx in range(self.config.n_boundary - sum(per_face)):
                per_face[idx % n_faces] += 1

            t_parts: list[Tensor] = []
            x_parts: list[Tensor] = []
            for face, n_f in enumerate(per_face):
                if n_f == 0:
                    continue
                axis, upper = divmod(face, 2)
                t_f = (
                    torch.rand(n_f, device=self.config.device) * (t_max - t_min)
                    + t_min
                )
                coords: list[Tensor] = []
                for dim in range(spatial_dim):
                    lo, hi = x_bounds[2 * dim], x_bounds[2 * dim + 1]
                    if dim == axis:
                        pinned = hi if upper else lo
                        coords.append(
                            torch.full((n_f,), pinned, device=self.config.device)
                        )
                    else:
                        coords.append(
                            torch.rand(n_f, device=self.config.device) * (hi - lo) + lo
                        )
                t_parts.append(t_f)
                x_parts.append(torch.stack(coords, dim=-1))

            return torch.cat(t_parts), torch.cat(x_parts, dim=0)

        x_min, x_max, y_min, y_max = x_bounds
        edge_count = max(1, self.config.n_boundary // 4)
        remainder = self.config.n_boundary - 4 * edge_count
        per_edge = [edge_count, edge_count, edge_count, edge_count]
        for idx in range(remainder):
            per_edge[idx % 4] += 1

        t_edges: list[Tensor] = []
        x_edges: list[Tensor] = []

        n0 = per_edge[0]
        t0 = torch.rand(n0, device=self.config.device) * (t_max - t_min) + t_min
        y0 = torch.rand(n0, device=self.config.device) * (y_max - y_min) + y_min
        x0 = torch.stack([torch.full((n0,), x_min, device=self.config.device), y0], dim=-1)
        t_edges.append(t0)
        x_edges.append(x0)

        n1 = per_edge[1]
        t1 = torch.rand(n1, device=self.config.device) * (t_max - t_min) + t_min
        y1 = torch.rand(n1, device=self.config.device) * (y_max - y_min) + y_min
        x1 = torch.stack([torch.full((n1,), x_max, device=self.config.device), y1], dim=-1)
        t_edges.append(t1)
        x_edges.append(x1)

        n2 = per_edge[2]
        t2 = torch.rand(n2, device=self.config.device) * (t_max - t_min) + t_min
        x2_coord = torch.rand(n2, device=self.config.device) * (x_max - x_min) + x_min
        x2 = torch.stack([x2_coord, torch.full((n2,), y_min, device=self.config.device)], dim=-1)
        t_edges.append(t2)
        x_edges.append(x2)

        n3 = per_edge[3]
        t3 = torch.rand(n3, device=self.config.device) * (t_max - t_min) + t_min
        x3_coord = torch.rand(n3, device=self.config.device) * (x_max - x_min) + x_min
        x3 = torch.stack([x3_coord, torch.full((n3,), y_max, device=self.config.device)], dim=-1)
        t_edges.append(t3)
        x_edges.append(x3)

        return torch.cat(t_edges, dim=0), torch.cat(x_edges, dim=0)

    def _generate_initial_points(
        self,
        pde_problem: PDEProblem
    ) -> tuple[Tensor, Tensor]:
        """Generate points at initial time (t = 0).

        Args:
            pde_problem: The PDE problem defining the domain.

        Returns:
            Tuple of (t, x) coordinate tensors at t=0.
        """
        _, x_bounds_raw = pde_problem.domain_bounds
        x_bounds = cast(tuple[float, ...], x_bounds_raw)
        t = torch.zeros(self.config.n_initial, device=self.config.device)
        x = self._sample_spatial_points(self.config.n_initial, x_bounds)

        return t, x

    def _infer_spatial_dim(self, x_bounds: tuple[float, ...]) -> int:
        if len(x_bounds) % 2 != 0:
            raise ValueError(f"Invalid spatial bounds length: {len(x_bounds)}")
        return len(x_bounds) // 2

    def _sample_spatial_points(self, num_points: int, x_bounds: tuple[float, ...]) -> Tensor:
        spatial_dim = self._infer_spatial_dim(x_bounds)
        if spatial_dim == 1:
            x_min, x_max = x_bounds[0], x_bounds[1]
            return torch.rand(num_points, device=self.config.device) * (x_max - x_min) + x_min

        components: list[Tensor] = []
        for dim in range(spatial_dim):
            x_min = x_bounds[2 * dim]
            x_max = x_bounds[2 * dim + 1]
            values = torch.rand(num_points, device=self.config.device) * (x_max - x_min) + x_min
            components.append(values)
        return torch.stack(components, dim=-1)

    def _build_inputs(self, t: Tensor, x: Tensor) -> Tensor:
        t_col = t.unsqueeze(-1) if t.dim() == 1 else t
        x_col = x.unsqueeze(-1) if x.dim() == 1 else x
        return torch.cat([t_col, x_col], dim=-1)

    def run_single(
        self,
        model: nn.Module,
        pde_problem: PDEProblem,
        config: BenchmarkConfig | None = None,
    ) -> dict[str, Any]:
        """Run comprehensive benchmark on a single model.

        Args:
            model: The PINN model to evaluate.
            pde_problem: The PDE problem instance.
            config: Optional override configuration.

        Returns:
            Dictionary containing all computed metrics:
                - Solution error metrics (relative_l2, mae, mse, max_error)
                - Physics violation metrics (pde_residual_norm, bc_violation, ic_violation)
                - Physics consistency metrics (if enabled)
        """
        cfg = config or self.config
        device = torch.device(cfg.device)
        model = model.to(device)
        model.eval()

        results: dict[str, Any] = {}
        residuals: Tensor | None = None

        # Generate evaluation points
        t_colloc, x_colloc = self._generate_collocation_points(pde_problem)
        t_bc, x_bc = self._generate_boundary_points(pde_problem)
        t_ic, x_ic = self._generate_initial_points(pde_problem)
        u_pred_colloc = torch.zeros_like(t_colloc)

        # Compute PDE residual (requires gradients)
        if 'pde_residual_norm' in cfg.metric_names:
            # Prepare inputs with gradients for PDE residual
            t_grad = t_colloc.clone().requires_grad_(True)
            x_grad = x_colloc.clone().requires_grad_(True)
            inputs_grad = self._build_inputs(t_grad, x_grad)
            u_pred_grad = model(inputs_grad).squeeze(-1)
            residuals_value = cast(Tensor, pde_problem.residual(u_pred_grad, t_grad, x_grad))
            residuals = residuals_value
            results['pde_residual_norm'] = pde_residual_norm(residuals_value.detach()).item()
            u_pred_colloc = u_pred_grad.detach()
        else:
            # Compute predictions without gradients
            inputs_colloc = self._build_inputs(t_colloc, x_colloc)
            u_pred_colloc = model(inputs_colloc).squeeze(-1)

        # Compute other metrics without gradients
        with torch.no_grad():
            # Compute exact solution at collocation points
            u_exact_colloc = pde_problem.exact_solution(t_colloc, x_colloc)

            if u_exact_colloc is not None:
                # Solution error metrics
                if 'relative_l2_error' in cfg.metric_names:
                    results['relative_l2_error'] = relative_l2_error(
                        u_pred_colloc, u_exact_colloc
                    ).item()
                if 'mse' in cfg.metric_names:
                    results['mse'] = mse(u_pred_colloc, u_exact_colloc).item()
                if 'mae' in cfg.metric_names:
                    results['mae'] = mae(u_pred_colloc, u_exact_colloc).item()
                if 'max_error' in cfg.metric_names:
                    results['max_error'] = max_error(u_pred_colloc, u_exact_colloc).item()

            # Boundary condition evaluation
            if 'bc_violation' in cfg.metric_names:
                inputs_bc = self._build_inputs(t_bc, x_bc)
                u_pred_bc = model(inputs_bc).squeeze(-1)
                u_true_bc = pde_problem.boundary_condition(t_bc, x_bc)
                results['bc_violation'] = bc_violation(u_pred_bc, u_true_bc).item()

            # Initial condition evaluation
            if 'ic_violation' in cfg.metric_names:
                inputs_ic = self._build_inputs(t_ic, x_ic)
                u_pred_ic = model(inputs_ic).squeeze(-1)
                u_true_ic = pde_problem.initial_condition(x_ic)
                results['ic_violation'] = ic_violation(u_pred_ic, u_true_ic).item()

            # Physics consistency checks
            spatial_bounds = cast(tuple[float, ...], pde_problem.domain_bounds[1])
            if cfg.include_physics_consistency and self._infer_spatial_dim(spatial_bounds) == 1:
                energy_errors = self.physics_checker.check_energy_conservation(
                    u_pred_colloc, t_colloc, x_colloc, pde_problem
                )
                results['energy_conservation_mean_error'] = energy_errors.mean().item()

                if 'pde_residual_norm' in cfg.metric_names and residuals is not None:
                    residual_dist = self.physics_checker.check_residual_distribution(
                        residuals.detach()
                    )
                    results['residual_mean'] = residual_dist['mean'].item()
                    results['residual_std'] = residual_dist['std'].item()

        return results

    def run_comparison(
        self,
        models: dict[str, nn.Module],
        pde_problem: PDEProblem,
        config: BenchmarkConfig | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Run comparison benchmark across multiple models.

        Args:
            models: Dictionary mapping model names to model instances.
            pde_problem: The PDE problem instance.
            config: Optional override configuration.

        Returns:
            Dictionary mapping model names to their metric dictionaries.
            Each inner dictionary contains all computed metrics for that model.
        """
        comparison_results: dict[str, dict[str, Any]] = {}

        for name, model in models.items():
            comparison_results[name] = self.run_single(model, pde_problem, config)

        return comparison_results

    @staticmethod
    def format_comparison_table(
        comparison_results: dict[str, dict[str, Any]]
    ) -> str:
        """Format comparison results as a readable table.

        Args:
            comparison_results: Results from run_comparison.

        Returns:
            Formatted string table of results.
        """
        if not comparison_results:
            return "No results to display."

        # Get all metric names
        metric_names_set: set[str] = set()
        for results in comparison_results.values():
            metric_names_set.update(results.keys())
        metric_names = sorted(metric_names_set)

        # Calculate column widths
        name_width = max(len(name) for name in comparison_results)
        name_width = max(name_width, len("Model"))

        metric_widths = {m: max(len(m), 12) for m in metric_names}

        # Build header
        header = f"{'Model':<{name_width}}"
        for metric in metric_names:
            header += f" | {metric:>{metric_widths[metric]}}"

        # Build separator
        separator = "-" * name_width
        for metric in metric_names:
            separator += "-+-" + "-" * metric_widths[metric]

        # Build rows
        rows = []
        for name, results in comparison_results.items():
            row = f"{name:<{name_width}}"
            for metric in metric_names:
                value = results.get(metric, float('nan'))
                if isinstance(value, float):
                    row += f" | {value:>{metric_widths[metric]}.6e}"
                else:
                    row += f" | {str(value):>{metric_widths[metric]}}"
            rows.append(row)

        # Combine into table
        table = f"{header}\n{separator}\n" + "\n".join(rows)
        return table
