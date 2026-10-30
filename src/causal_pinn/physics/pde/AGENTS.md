<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-07-24 | Updated: 2026-07-24 -->

# pde

## Purpose
Defines the abstract `PDEProblem` interface and thirteen concrete PDE problems used throughout the framework. Each concrete class encapsulates a governing equation's residual, boundary/initial conditions, domain bounds, and (where one exists) an exact analytical solution. The `residual(u, t, x)` method is the single hook that couples a PDE to the training losses in `causal_pinn.physics.losses` — it is called by `PhysicsLoss`, `CausalLoss`, and related losses without any other dependency on the PDE class.

## Key Files
| File | Description |
|------|-------------|
| `base.py` | `PDEProblem(ABC)`: abstract base declaring `domain_bounds` (property), `residual(u, t, x)`, `boundary_condition(t, x)`, `initial_condition(x)` as abstract methods, plus a default `exact_solution(t, x) -> None` and static helpers `_ensure_tensor` and `_compute_derivative` (a convenience `torch.autograd.grad` wrapper with `allow_unused=True`, returning zeros if the graph is disconnected). |
| `heat_1d.py` | `Heat1D`: linear 1D heat/diffusion equation `u_t = alpha * u_xx` on `[0, L] x [0, T]`, Dirichlet BCs `u=0`, exact solution `sin(pi x/L) exp(-alpha (pi/L)^2 t)`. |
| `burgers_1d.py` | `Burgers1D`: nonlinear 1D viscous Burgers equation `u_t + u u_x = nu u_xx`, Dirichlet BCs `u=0`, IC `-sin(pi x/L)`, no exact solution (shock-forming, canonical hard case for causal training). |
| `burgers_2d.py` | `Burgers2D`: coupled 2D Burgers system for velocity field `(u, v)`; `residual` returns a `(batch, 2)` stacked tensor (both component residuals); Dirichlet BCs `u=v=0`; smooth sinusoidal IC; no exact solution. |
| `advection_diffusion_1d.py` | `AdvectionDiffusion1D`: linear `u_t + c u_x = alpha u_xx`; periodic BC implemented by evaluating `exact_solution` at the boundary; exact traveling-wave-with-decay solution `sin(k(x-ct)) exp(-alpha k^2 t)`. |
| `allen_cahn_1d.py` | `AllenCahn1D`: `u_t = eps^2 u_xx + u - u^3`. Two modes: default (Dirichlet `u=0`, IC `sin(pi x/L)`, no exact solution) or, when `front_center` is set, "front mode" using the exact stationary tanh-front `-tanh((x-x0)/(sqrt(2) eps))` as IC/BC/exact_solution. Uses `@override` decorators and `cast()` for autograd-grad `Optional` results. |
| `kdv_1d.py` | `KdV1D`: 1D Korteweg-de Vries equation `u_t + u u_x + mu u_xxx = 0` (third-order dispersion term, requires an extra autograd pass for `u_xxx`). Default mode: Dirichlet `u=0`, IC `-sin(pi x/L)`. Optional "soliton mode" (`soliton_amplitude` set): uses the exact one-soliton solution `3c sech^2(kappa(x - ct - x0))` as IC/BC/exact_solution, with `c = amplitude/3`, `kappa = sqrt(c/(4 mu))`. |
| `reaction_diffusion_1d.py` | `ReactionDiffusion1D`: 1D Fisher-KPP equation `u_t = D u_xx + r u (1-u)`, asymmetric Dirichlet BCs (`u(0,t)=1`, `u(L,t)=0`, via `torch.where` on `x < L/2`), smooth-step IC `0.5(1 - tanh(k(x - L/4)))`; no exact solution. Validates `diffusion > 0` and `reaction_rate >= 0` in `__init__`. |
| `wave_2d.py` | `Wave2D`: second-order-in-time 2D wave equation `u_tt = c^2 (u_xx + u_yy)`; residual requires two temporal autograd passes (`u_t` then `u_tt`) plus the 2D Laplacian; Dirichlet BCs on all sides; exact standing-wave solution `sin(pi x/Lx) sin(pi y/Ly) cos(omega t)`. |
| `heat_3d.py` | `Heat3D`: 3D heat equation `u_t = alpha * laplacian(u)` on a box, spatial coordinates packed as `(batch, 3)`; homogeneous Dirichlet on all six faces; exact separable solution `sin(pi x) sin(pi y) sin(pi z) exp(-alpha pi^2 (Lx^-2+Ly^-2+Lz^-2) t)`. Exists to give the suite a `d=3` configuration with a ground truth. |
| `multiscale_heat_2d.py` | `MultiscaleHeat2D`: 2D heat equation carrying two Dirichlet eigenmodes whose wavenumbers differ by `scale_ratio` (default 4) and whose decay rates therefore differ by its square (16). Exact by superposition; exposes `coarse_decay_rate()` / `fine_decay_rate()` and a `_modes(x)` helper returning the two spatial profiles, which the B1 probe uses to score each scale separately. The suite's only scale-separated configuration. |
| `burgers_1d_exact.py` | `BurgersExact1D`: `Burgers1D` plus the Cole-Hopf closed-form reference solution, so the shock case can be scored. Kept as a separate class so `burgers1d`'s `has_exact_solution` router feature does not silently change. |
| `reaction_diffusion_1d_exact.py` | `ReactionDiffusionExact1D`: Fisher-KPP in travelling-wave mode, using the Ablowitz-Zeppetella profile `[1 + exp(k(x-ct))]^-2` valid only at the speed `c = 5 sqrt(rD/6)`. Separate class for the same reason as the Burgers variant. |
| `navier_stokes_2d.py` | `NavierStokes2D`: 2D Navier-Stokes via Taylor-Green vortex, formulated as *linear* vorticity-transport residual `omega_t + vel_x omega_x + vel_y omega_y - nu laplacian(omega)`, where the advecting velocity field `(vel_x, vel_y)` is the *known exact* Taylor-Green solution (not learned) evaluated inline inside `residual`. Also exposes `exact_velocity(t, x)` and `exact_pressure(t, x)` helper methods (not part of the `PDEProblem` abstract interface) for analysis/plotting. Default domain is `[0, 2*pi] x [0, 2*pi]`, the vortex's natural period. Validates `viscosity > 0`. |
| `__init__.py` | Re-exports `PDEProblem` and all 13 concrete classes: `Heat1D`, `Heat3D`, `AdvectionDiffusion1D`, `AllenCahn1D`, `Burgers1D`, `BurgersExact1D`, `KdV1D`, `MultiscaleHeat2D`, `NavierStokes2D`, `ReactionDiffusion1D`, `ReactionDiffusionExact1D`, `Burgers2D`, `Wave2D`. |

## For AI Agents

### Working In This Directory
- To add a new PDE: subclass `PDEProblem`, implement `domain_bounds` (property), `residual(u, t, x)`, `boundary_condition(t, x)`, `initial_condition(x)`, and optionally override `exact_solution(t, x)`. Register the new class in `__init__.py`'s imports/`__all__`.
- Invariant enforced across every `residual()` implementation: **do not reshape `t`/`x` before passing them into `torch.autograd.grad`** — only reshape/flatten the *derivative outputs*. Reshaping the leaf tensors themselves breaks the autograd graph. The 1D PDEs all follow the same idiom: `u_flat = u.reshape(-1)`, then compute `u_t`/`u_x`/`u_xx` (etc.) via chained `torch.autograd.grad(..., create_graph=True, retain_graph=True)` calls with `if <deriv> is None: <deriv> = torch.zeros_like(...)` guards (autograd returns `None` when a variable is disconnected from the graph).
- For 2D PDEs (`burgers_2d.py`, `wave_2d.py`, `navier_stokes_2d.py`), spatial gradients come back as a single `(batch, 2)` tensor from one `torch.autograd.grad(outputs=u, inputs=x, ...)` call; components are sliced (`grad_u[:, 0]`, `grad_u[:, 1]`) rather than computed with separate calls.
- If a new PDE should participate in the meta-router's PDE selection/routing (Paper A), it must also be registered in `PDE_FEATURE_REGISTRY` in `src/causal_pinn/training/meta_router.py` — this directory only defines the physics, not routing metadata.
- Newer files (`allen_cahn_1d.py`, `kdv_1d.py`) use `@override` decorators and `cast()` around the `Optional` autograd result before the `None`-guard; older files (`heat_1d.py`, `burgers_1d.py`, etc.) do not — match whichever style is closer to the file you're editing rather than mixing them.

### Testing Requirements
- `tests/unit/test_pde.py` — general/shared PDE behavior (likely parametrized across multiple PDE classes).
- `tests/unit/test_pde_allen_cahn.py`, `tests/unit/test_pde_kdv.py` — dedicated tests for the dual-mode (default/front, default/soliton) PDEs.
- `tests/unit/test_burgers1d.py`, `tests/unit/test_navier_stokes_2d.py`, `tests/unit/test_reaction_diffusion.py` — PDE-specific tests.
- `tests/unit/test_pde_heat3d.py`, `tests/unit/test_pde_multiscale_heat_2d.py`, `tests/unit/test_burgers_exact.py`, `tests/unit/test_reaction_diffusion_exact.py` — the newer PDEs. For any PDE with a closed form, the load-bearing test is that the residual vanishes on the exact solution *and* that near-miss controls (wrong decay rate, wrong wavenumber, rescaled field) leave a large residual; a zero-residual test alone passes trivially for a residual that is blind to the property under test.
- Run: `PYTHONPATH=src python -m pytest tests/unit/test_pde.py tests/unit/test_pde_allen_cahn.py tests/unit/test_pde_kdv.py tests/unit/test_burgers1d.py tests/unit/test_navier_stokes_2d.py tests/unit/test_reaction_diffusion.py -q`

### Common Patterns
- `domain_bounds` returns `((t_min, t_max), x_bounds)` where `x_bounds` is a 2-tuple `(x_min, x_max)` for 1D problems or a 4-tuple `(x_min, x_max, y_min, y_max)` for 2D problems.
- Physical coefficients (`viscosity`, `alpha`, `diffusion`, `reaction_rate`, etc.) are stored as plain public float attributes set in `__init__`, with validation (`raise ValueError`) only where an invalid value would cause a degenerate/undefined equation (e.g. `epsilon <= 0`, `dispersion <= 0`, `diffusion <= 0`, `viscosity <= 0`).
- "Dual mode" PDEs (`AllenCahn1D` front mode, `KdV1D` soliton mode) gate all of `boundary_condition`, `initial_condition`, and `exact_solution` on the same optional constructor parameter (`front_center`, `soliton_amplitude`), so all three stay consistent with whichever mode is active.
- PDEs without a closed-form solution (`Burgers1D`, `Burgers2D`, `ReactionDiffusion1D`, and default-mode `AllenCahn1D`/`KdV1D`) explicitly return `None` from `exact_solution` rather than raising, matching the base class default.

## Dependencies

### Internal
- `.base.PDEProblem` (all concrete classes import from the sibling `base.py`).

### External
- `torch` (`Tensor`, `torch.autograd.grad`) in every residual computation.
- `math` (`allen_cahn_1d.py`, `kdv_1d.py`, `navier_stokes_2d.py` — for `sqrt`, `pi` in soliton/front parameter derivation and default domain length).
- `typing.cast`, `typing.override` (`allen_cahn_1d.py`, `kdv_1d.py` only).

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
