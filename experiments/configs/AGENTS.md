<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-07-24 | Updated: 2026-07-24 -->

# configs

## Purpose
YAML configuration files documenting the canonical model architecture and 3-stage training protocol used in the paper(s). These files are the human-readable, authoritative record of the hyperparameters claimed in the manuscript text — they are not universally auto-loaded by every experiment script (most `experiments/scripts/**` scripts hardcode their own hyperparameters and record the active config in their output JSON's `config` field), so treat these YAMLs as the spec to check scripts against rather than as a guaranteed runtime input.

## Key Files
| File | Description |
|------|-------------|
| `default.yaml` | Default experiment configuration aligned with the general paper protocol: PINN `[64,64,64,64]` + GELU, FNO (modes=8, width=32, 4 layers), Stage 1/2/3 = 200/500/200 epochs, `causal_epsilon=1.0`, 10,000 collocation points, `device: cuda`. Usable via `python -m causal_pinn.training.trainer --config experiments/configs/default.yaml` |
| `paper_combined.yaml` | Authoritative config for the combined/long-form paper and the NeurIPS ED-track submission — same architecture as `default.yaml` but with the paper's exact sampling protocol (batch_size=128, 4,000 collocation points, 500 boundary points, `causal_epsilon=10.0`, `output_dir: experiments/results/paper_combined`). Do not edit without cross-checking the paper text |

## For AI Agents

### Working In This Directory
- These are data files, not code — edits here have no effect unless a script or `causal_pinn.training.trainer --config` explicitly loads them. Changing a value does not retroactively change any experiment script's hardcoded constants.
- `paper_combined.yaml`'s header explicitly warns: "Do not edit without cross-updating the paper text" — treat any edit to `training:` values here as a claim-affecting change requiring the same scrutiny as changing a script's hardcoded hyperparameters.
- If you add a new paper-specific config, follow the existing two-part shape: `model:` (PINN dims/activation, FNO modes/width/layers) and `training:` (per-stage epochs/lr, `lambda_physics`, `lambda_data`, `causal_epsilon`, batch/collocation/boundary point counts, `seed`, `device`), plus top-level `evaluation:` and `output_dir:`.

### Testing Requirements
No tests load these files directly. `make lint` (ruff) does not lint YAML. Validate a new/edited config by hand-inspecting it against the paper's Sec. 4.1 (architecture) / 4.2 (training protocol) description, or by loading it through `causal_pinn.utils.config.TrainingConfig` if a script consumes it.

### Common Patterns
- Both files share the same `model:`/`training:`/`evaluation:`/`output_dir:` top-level structure; keep new configs consistent with this shape so any future config-loading code can treat them uniformly.
- `output_dir:` points at the `experiments/results/...` subtree where that config's runs should land — align it with the `--output-dir` values used in the corresponding `experiments/scripts/**` invocations.

## Dependencies

### Internal
- `causal_pinn.utils.config.TrainingConfig` — the dataclass these YAMLs' `training:` section maps onto (library defaults may differ; these YAMLs are the paper-authoritative superset).
- `causal_pinn.training.trainer` — the `python -m` entrypoint referenced by `default.yaml`'s usage comment.

### External
- None directly (plain YAML); consumed via PyYAML/OmegaConf if/when a script loads them.

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
