from typing import Tuple

import torch
import torch.nn as nn

from .model import CNN


class GroundModel(nn.Module):
    def __init__(self, num_cells: int, embed_dim: int = 256, backbone: str = "resnet18", pretrained_backbone: bool = True, in_chans: int = 3, dropout: float = 0.1):
        super().__init__()
        self.net = CNN(
            num_cells=num_cells,
            embed_dim=embed_dim,
            backbone=backbone,
            pretrained_backbone=pretrained_backbone,
            in_chans=in_chans,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.net(x)

    @staticmethod
    def from_config(cfg) -> "GroundModel":
        return GroundModel(
            num_cells=int(cfg["num_cells"]),
            embed_dim=int(cfg.get("embed_dim", 256)),
            backbone=str(cfg.get("backbone", "resnet18")),
            pretrained_backbone=bool(cfg.get("pretrained_backbone", True)),
            in_chans=int(cfg.get("in_chans", 3)),
            dropout=float(cfg.get("dropout", 0.1))
        )
