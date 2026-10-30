from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from .base import NeuralOperator


def _build_mlp(input_dim: int, output_dim: int, num_layers: int) -> nn.Sequential:
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1")

    if num_layers == 1:
        return nn.Sequential(nn.Linear(input_dim, output_dim))

    layers: list[nn.Module] = [nn.Linear(input_dim, output_dim), nn.GELU()]
    for _ in range(num_layers - 2):
        layers.extend([nn.Linear(output_dim, output_dim), nn.GELU()])
    layers.append(nn.Linear(output_dim, output_dim))
    return nn.Sequential(*layers)


class DeepONet(NeuralOperator):
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        branch_dim: int = 64,
        trunk_dim: int = 64,
        num_layers: int = 3,
        dim: int = 1,
    ) -> None:
        super().__init__()

        if dim != 1:
            raise ValueError("DeepONet currently supports only dim=1")
        if branch_dim != trunk_dim:
            raise ValueError("branch_dim and trunk_dim must be equal")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.branch_dim = branch_dim
        self.trunk_dim = trunk_dim
        self.num_layers = num_layers
        self.dim = dim

        self.branch_net = _build_mlp(in_channels, branch_dim, num_layers)
        self.trunk_net = _build_mlp(in_channels, trunk_dim, num_layers)
        self.output_layer = nn.Linear(branch_dim, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        point_features = x.transpose(1, 2)

        branch_input = x.mean(dim=-1)

        branch_latent = self.branch_net(branch_input)
        trunk_latent = self.trunk_net(point_features)

        fused = trunk_latent * branch_latent.unsqueeze(1)
        out = self.output_layer(fused)

        return out.transpose(1, 2)  # type: ignore[no-any-return]

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
