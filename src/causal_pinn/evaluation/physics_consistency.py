"""Physics consistency verification for PINN solutions.

This module provides tools for verifying that PINN solutions satisfy
physical conservation laws and analyzing residual distributions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

if TYPE_CHECKING:
    from causal_pinn.physics.pde.base import PDEProblem


class PhysicsConsistencyChecker:
    """Checker for physical consistency of PINN solutions.

    This class provides methods to verify that neural network solutions
    satisfy physical conservation laws such as energy and mass conservation.

    Attributes:
        device: Torch device for computations.
    """

    def __init__(self, device: torch.device | str | None = None) -> None:
        """Initialize the physics consistency checker.

        Args:
            device: Torch device for computations. Defaults to 'cpu'.
        """
        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(device) if isinstance(device, str) else device

    def check_energy_conservation(
        self,
        u_pred: Tensor,
        t: Tensor,
        x: Tensor,
        pde_problem: PDEProblem,
    ) -> Tensor:
        """Check energy conservation over time for 1D problems.

        For diffusion-type problems, energy (L2 norm squared) should
        decrease or remain constant over time due to dissipation.

        Computes the L2 energy E(t) = integral(u^2 dx) at each time point
        and returns the error relative to a reference (typically t=0).

        Args:
            u_pred: Predicted solution tensor of shape (N,) or (N, 1).
            t: Time coordinates tensor of shape (N,) or (N, 1).
            x: Spatial coordinates tensor of shape (N,) or (N, 1).
            pde_problem: The PDE problem instance for domain information.

        Returns:
            Tensor of energy errors at each unique time point, representing
            deviation from energy conservation. Shape: (num_unique_times,).
        """
        u_pred = u_pred.reshape(-1).to(self.device)
        t = t.reshape(-1).to(self.device)
        x = x.reshape(-1).to(self.device)

        # Get unique time points
        unique_times = torch.unique(t)

        if len(unique_times) < 2:
            return torch.zeros(1, device=self.device)

        # Get domain bounds for integration
        _, x_bounds = pde_problem.domain_bounds

        # Compute energy at each time point
        energies = []
        for time_val in unique_times:
            mask = torch.isclose(t, time_val)
            u_t = u_pred[mask]
            x_t = x[mask]

            if len(u_t) < 2:
                energies.append(torch.tensor(0.0, device=self.device))
                continue

            # Sort by spatial coordinate for proper integration
            sort_idx = torch.argsort(x_t)
            x_sorted = x_t[sort_idx]
            u_sorted = u_t[sort_idx]

            # Approximate L2 energy using trapezoidal rule
            dx = x_sorted[1:] - x_sorted[:-1]
            u_avg_sq = (u_sorted[1:] ** 2 + u_sorted[:-1] ** 2) / 2
            energy = torch.sum(u_avg_sq * dx)
            energies.append(energy)

        # Stack energies
        energies = torch.stack(energies)

        # Compute relative deviation from initial energy
        if energies[0] > 0:
            energy_error = torch.abs(energies - energies[0]) / energies[0]
        else:
            energy_error = torch.zeros_like(energies)

        return energy_error

    def check_mass_conservation(
        self,
        u_pred: Tensor,
        t: Tensor,
        x: Tensor,
        pde_problem: PDEProblem,
    ) -> Tensor:
        """Check mass conservation over time for 1D problems.

        For conservative systems, total mass (integral of u) should
        remain constant over time.

        Computes the mass M(t) = integral(u dx) at each time point
        and returns the error relative to a reference (typically t=0).

        Args:
            u_pred: Predicted solution tensor of shape (N,) or (N, 1).
            t: Time coordinates tensor of shape (N,) or (N, 1).
            x: Spatial coordinates tensor of shape (N,) or (N, 1).
            pde_problem: The PDE problem instance for domain information.

        Returns:
            Tensor of mass errors at each unique time point, representing
            deviation from mass conservation. Shape: (num_unique_times,).
        """
        u_pred = u_pred.reshape(-1).to(self.device)
        t = t.reshape(-1).to(self.device)
        x = x.reshape(-1).to(self.device)

        # Get unique time points
        unique_times = torch.unique(t)

        if len(unique_times) < 2:
            return torch.zeros(1, device=self.device)

        # Compute mass at each time point
        masses = []
        for time_val in unique_times:
            mask = torch.isclose(t, time_val)
            u_t = u_pred[mask]
            x_t = x[mask]

            if len(u_t) < 2:
                masses.append(torch.tensor(0.0, device=self.device))
                continue

            # Sort by spatial coordinate for proper integration
            sort_idx = torch.argsort(x_t)
            x_sorted = x_t[sort_idx]
            u_sorted = u_t[sort_idx]

            # Approximate mass using trapezoidal rule
            dx = x_sorted[1:] - x_sorted[:-1]
            u_avg = (u_sorted[1:] + u_sorted[:-1]) / 2
            mass = torch.sum(u_avg * dx)
            masses.append(mass)

        # Stack masses
        masses = torch.stack(masses)

        # Compute relative deviation from initial mass
        if torch.abs(masses[0]) > 1e-10:
            mass_error = torch.abs(masses - masses[0]) / torch.abs(masses[0])
        else:
            mass_error = torch.zeros_like(masses)

        return mass_error

    def check_residual_distribution(
        self,
        residuals: Tensor,
    ) -> dict[str, Any]:
        """Analyze the distribution of PDE residuals.

        Computes statistical measures of the residual distribution
        including mean, standard deviation, maximum, and percentiles.

        Args:
            residuals: Tensor of PDE residual values.

        Returns:
            Dictionary containing:
                - 'mean': Mean of residuals (scalar tensor)
                - 'std': Standard deviation of residuals (scalar tensor)
                - 'max': Maximum absolute residual (scalar tensor)
                - 'percentiles': Dict with 25th, 50th, 75th, 95th percentiles
                - 'min': Minimum residual (scalar tensor)
        """
        residuals = residuals.reshape(-1).to(self.device)

        # Compute basic statistics
        mean = torch.mean(residuals)
        std = torch.std(residuals) if len(residuals) > 1 else torch.tensor(0.0, device=self.device)
        max_val = torch.max(torch.abs(residuals))
        min_val = torch.min(residuals)

        # Compute percentiles
        percentiles = {}
        for p in [25, 50, 75, 95]:
            percentiles[str(p)] = torch.quantile(residuals.abs(), p / 100.0)

        return {
            'mean': mean,
            'std': std,
            'max': max_val,
            'min': min_val,
            'percentiles': percentiles,
        }
