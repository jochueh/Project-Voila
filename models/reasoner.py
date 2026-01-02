from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DEMEncoder(nn.Module):
    def __init__(self, in_chans: int = 1, dem_embed_dim: int = 128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_chans, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(64, dem_embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.enc(x).flatten(1)
        z = self.fc(f)
        return F.normalize(z, dim=1)


class Reasoner(nn.Module):
    def __init__(self, img_embed_dim: int, num_cells: int, dem_embed_dim: int = 128, hidden: int = 256, k: int = 5):
        super().__init__()
        self.k = k
        self.dem_encoder = DEMEncoder(in_chans=1, dem_embed_dim=dem_embed_dim)
        self.fuse = nn.Sequential(
            nn.Linear(img_embed_dim + dem_embed_dim, hidden),
            nn.ReLU(inplace=True)
        )
        self.delta_logits = nn.Linear(hidden, num_cells)
        self.delta_offsets = nn.Linear(hidden, 2)

        nn.init.trunc_normal_(self.delta_logits.weight, std=0.02)
        nn.init.zeros_(self.delta_logits.bias)
        nn.init.trunc_normal_(self.delta_offsets.weight, std=0.02)
        nn.init.zeros_(self.delta_offsets.bias)

    def forward(
        self,
        img_embed: torch.Tensor,
        dem_patches: torch.Tensor,
        topk_idx: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, k, h, w = dem_patches.shape
        dem_patches = dem_patches.view(b * k, 1, h, w)
        dem_z = self.dem_encoder(dem_patches).view(b, k, -1)

        img_z = F.normalize(img_embed, dim=1).unsqueeze(1).expand(-1, k, -1)
        fused = torch.cat([img_z, dem_z], dim=2)
        fused = self.fuse(fused)

        dlogits = self.delta_logits(fused).mean(dim=1)
        doff = torch.tanh(self.delta_offsets(fused).mean(dim=1))
        return dlogits, doff
