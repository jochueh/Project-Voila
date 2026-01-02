import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import SmoothL1Loss, CrossEntropyLoss


def haversine_km(lat1: Tensor, lon1: Tensor, lat2: Tensor, lon2: Tensor) -> Tensor:
    r = 6371.0
    dlat = torch.deg2rad(lat2 - lat1)
    dlon = torch.deg2rad(lon2 - lon1)
    a = torch.sin(dlat * 0.5).pow(2) + torch.cos(torch.deg2rad(lat1)) * torch.cos(torch.deg2rad(lat2)) * torch.sin(dlon * 0.5).pow(2)
    c = 2.0 * torch.atan2(torch.sqrt(a), torch.sqrt(1.0 - a))
    return r * c


def grid_targets(lat: Tensor, lon: Tensor, lat_bins: int, lon_bins: int) -> Tuple[Tensor, Tensor]:
    lat_n = (lat + 90.0) / 180.0 * lat_bins
    lon_n = (lon + 180.0) / 360.0 * lon_bins
    lat_bin = torch.clamp(lat_n.floor().long(), 0, lat_bins - 1)
    lon_bin = torch.clamp(lon_n.floor().long(), 0, lon_bins - 1)
    cell = lat_bin * lon_bins + lon_bin
    frac_lat = lat_n - lat_n.floor()
    frac_lon = lon_n - lon_n.floor()
    offs = torch.stack([frac_lat * 2.0 - 1.0, frac_lon * 2.0 - 1.0], dim=1)
    return cell, offs


def triplet_loss(anchor: Tensor, positive: Tensor, negative: Tensor, margin: float = 0.2) -> Tensor:
    d_pos = (anchor - positive).pow(2).sum(dim=1)
    d_neg = (anchor - negative).pow(2).sum(dim=1)
    return F.relu(d_pos - d_neg + margin).mean()


def batch_hard_triplets(embeds: Tensor, lat: Tensor, lon: Tensor) -> Tuple[Tensor, Tensor]:
    b = embeds.size(0)
    lat1 = lat.view(b, 1).repeat(1, b)
    lon1 = lon.view(b, 1).repeat(1, b)
    lat2 = lat.view(1, b).repeat(b, 1)
    lon2 = lon.view(1, b).repeat(b, 1)
    d = haversine_km(lat1, lon1, lat2, lon2)
    d = d + torch.eye(b, device=d.device) * 1e9
    pos_idx = torch.argmin(d, dim=1)
    neg_idx = torch.argmax(d, dim=1)
    pos = embeds[pos_idx]
    neg = embeds[neg_idx]
    return pos, neg


class LossBundle:
    def __init__(self, w_cls: float, w_reg: float, w_trip: float, w_rcls: float, w_rreg: float):
        self.ce = CrossEntropyLoss()
        self.l1 = SmoothL1Loss()
        self.w_cls = float(w_cls)
        self.w_reg = float(w_reg)
        self.w_trip = float(w_trip)
        self.w_rcls = float(w_rcls)
        self.w_rreg = float(w_rreg)

    def __call__(self, logits, offsets, embeds, cell_t, off_t, lat, lon, dlogits=None, doff=None):
        pos, neg = batch_hard_triplets(embeds, lat, lon)
        l_cls = self.ce(logits, cell_t)
        l_reg = self.l1(offsets, off_t)
        l_trip = triplet_loss(embeds, pos, neg)
        total = self.w_cls * l_cls + self.w_reg * l_reg + self.w_trip * l_trip
        extras = {"L_cls": l_cls.item(), "L_reg": l_reg.item(), "L_trip": l_trip.item()}
        if dlogits is not None and doff is not None:
            l_rcls = self.ce(dlogits, cell_t)
            l_rreg = self.l1(doff, off_t)
            total = total + self.w_rcls * l_rcls + self.w_rreg * l_rreg
            extras["L_rcls"] = l_rcls.item()
            extras["L_rreg"] = l_rreg.item()
        return total, extras
