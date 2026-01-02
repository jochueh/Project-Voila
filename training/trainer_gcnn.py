import os
import time
import yaml
import math
import argparse
from typing import Dict, Tuple, Optional

import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms as T
from pathlib import Path


from models.ground_cnn import GroundModel
from training.losses import LossBundle
from training.datasets import _grid_encode


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed_all(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_yaml(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _make_run_dir(base_dir: str) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")

    def try_make(p: str) -> str:
        os.makedirs(p, exist_ok=True)
        run = os.path.join(p, stamp)
        os.makedirs(run, exist_ok=True)
        return run

    try:
        return try_make(base_dir)
    except PermissionError:
        fallback = os.path.join(os.getcwd(), "checkpoints")
        print(f"[warn] cannot write to {base_dir}; falling back to {fallback}")
        return try_make(fallback)


def _default_img_tf(size_hw: Tuple[int, int]) -> T.Compose:
    h, w = size_hw
    return T.Compose([
        T.Resize((h, w), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class GroundOnlyDataset(Dataset):
    def __init__(self, root_processed: str, lat_bins: int, lon_bins: int, img_size_hw: Tuple[int, int],
                 sample_limit: Optional[int] = None, shuffle: bool = True,
                 images_subdir: str = "images", metadata_csv: str = "metadata.csv"):
        self.lat_bins = int(lat_bins)
        self.lon_bins = int(lon_bins)
        self.img_tf = _default_img_tf(img_size_hw)

        meta_csv_path = os.path.join(root_processed, metadata_csv)
        img_root = os.path.join(root_processed, images_subdir)
        if not os.path.isfile(meta_csv_path):
            raise FileNotFoundError(f"metadata.csv not found at {meta_csv_path}")
        if not os.path.isdir(img_root):
            raise FileNotFoundError(f"images directory not found at {img_root}")

        df = pd.read_csv(meta_csv_path, dtype={"id": "string"}, low_memory=False)

        req_cols = {"id", "latitude", "longitude"}
        if not req_cols.issubset(df.columns):
            raise ValueError(f"metadata.csv must have columns {req_cols}, got {set(df.columns)}")

        if sample_limit is not None and sample_limit > 0 and len(df) > sample_limit:
            df = df.sample(n=sample_limit, random_state=42).reset_index(drop=True)

        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

        # Build id -> path mapping by scanning images once
        id2path = {}
        for p in Path(img_root).rglob("*"):
            if not p.is_file():
                continue
            suf = p.suffix.lower()
            if suf not in exts:
                continue
            stem = p.stem
            if stem not in id2path:
                id2path[stem] = str(p)

        recs = []
        for _, r in df.iterrows():
            k = str(r["id"]).strip()
            if not k or k == "nan":
                continue
            pth = id2path.get(k)
            if pth is None:
                continue
            recs.append((pth, float(r["latitude"]), float(r["longitude"])))

        if shuffle and len(recs) > 1:
            rng = np.random.default_rng(42)
            rng.shuffle(recs)

        self.records = recs


    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        p, lat, lon = self.records[idx]
        img = Image.open(p).convert("RGB")
        img_t = self.img_tf(img)

        _, _, cid, _, frac_lat, frac_lon = _grid_encode(lat, lon, self.lat_bins, self.lon_bins)
        off = np.array([frac_lat * 2.0 - 1.0, frac_lon * 2.0 - 1.0], dtype=np.float32)

        return {
            "image": img_t,
            "lat": torch.tensor(lat, dtype=torch.float32),
            "lon": torch.tensor(lon, dtype=torch.float32),
            "cell": torch.tensor(int(cid), dtype=torch.long),
            "offset": torch.from_numpy(off),
        }


def train_one_epoch(epoch, net, loader, losses, optim, sched, scaler, device, log_every=100):
    net.train()
    total = 0.0
    count = 0
    for i, batch in enumerate(loader, 1):
        img = batch["image"].to(device, non_blocking=True)
        lat = batch["lat"].to(device, non_blocking=True)
        lon = batch["lon"].to(device, non_blocking=True)
        cell_t = batch["cell"].to(device, non_blocking=True)
        off_t = batch["offset"].to(device, non_blocking=True)

        optim.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits, offsets, embeds = net(img)
            loss, extras = losses(logits, offsets, embeds, cell_t, off_t, lat, lon, None, None)

        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        if sched is not None:
            sched.step()

        total += loss.item() * img.size(0)
        count += img.size(0)

        if i % log_every == 0:
            avg = total / max(1, count)
            print(f"Epoch {epoch} Iter {i}/{len(loader)} AvgLoss {avg:.4f} "
                  f"L_cls {extras.get('L_cls', 0):.4f} L_reg {extras.get('L_reg', 0):.4f} L_trip {extras.get('L_trip', 0):.4f}")

    return total / max(1, count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-cfg", required=True)
    parser.add_argument("--grid-cfg", required=True)
    parser.add_argument("--model-cfg", required=True)
    parser.add_argument("--train-cfg", required=True)
    parser.add_argument("--runs-dir", default="experiments/runs/checkpoints")
    args = parser.parse_args()

    data_cfg = _load_yaml(args.data_cfg)
    grid_cfg = _load_yaml(args.grid_cfg)
    model_cfg = _load_yaml(args.model_cfg)
    train_cfg = _load_yaml(args.train_cfg)

    seed = int(train_cfg.get("seed", 42))
    _seed_all(seed)

    root_processed = str(data_cfg.get("processed_root") or data_cfg.get("root"))
    images_subdir = str(data_cfg.get("images_subdir", "images"))
    metadata_csv = str(data_cfg.get("metadata_csv", "metadata.csv"))

    lat_bins = int(grid_cfg["lat_bins"])
    lon_bins = int(grid_cfg["lon_bins"])
    img_h = int(model_cfg.get("img_h", 224))
    img_w = int(model_cfg.get("img_w", 224))

    batch_size = int(train_cfg.get("batch_size", 64))
    workers = int(train_cfg.get("workers", 6))
    epochs = int(train_cfg.get("epochs", 30))

    ds = GroundOnlyDataset(
        root_processed=root_processed,
        lat_bins=lat_bins,
        lon_bins=lon_bins,
        img_size_hw=(img_h, img_w),
        sample_limit=train_cfg.get("sample_limit", None),
        shuffle=True,
        images_subdir=images_subdir,
        metadata_csv=metadata_csv,
    )
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if workers > 0 else False,
        prefetch_factor=2 if workers > 0 else None,
    )

    device = _device()
    num_cells = lat_bins * lon_bins
    gmodel = GroundModel.from_config({
        "num_cells": num_cells,
        "embed_dim": model_cfg.get("embed_dim", 256),
        "backbone": model_cfg.get("backbone", "resnet18"),
        "pretrained_backbone": model_cfg.get("pretrained_backbone", True),
        "in_chans": 3,
        "dropout": model_cfg.get("dropout", 0.1),
    }).to(device)

    w_cls = float(train_cfg.get("w_cls", 1.0))
    w_reg = float(train_cfg.get("w_reg", 10.0))
    w_trip = float(train_cfg.get("w_trip", 1.0))
    losses = LossBundle(w_cls, w_reg, w_trip, w_rcls=0.0, w_rreg=0.0)

    lr = float(train_cfg.get("lr", 1e-4))
    wd = float(train_cfg.get("weight_decay", 0.05))
    optim = AdamW(list(gmodel.parameters()), lr=lr, weight_decay=wd)

    tmax = epochs * max(1, math.ceil(len(dl)))
    sched = CosineAnnealingLR(optim, T_max=tmax)

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    run_dir = _make_run_dir(args.runs_dir)
    for name, cfg in [("data.yaml", data_cfg), ("grid.yaml", grid_cfg), ("model.yaml", model_cfg), ("train.yaml", train_cfg)]:
        with open(os.path.join(run_dir, name), "w") as f:
            yaml.safe_dump(cfg, f)

    for ep in range(1, epochs + 1):
        avg = train_one_epoch(ep, gmodel, dl, losses, optim, sched, scaler, device, log_every=int(train_cfg.get("log_every", 100)))
        out_path = os.path.join(run_dir, f"weights_epoch{ep}.pth")
        torch.save({"model": gmodel.state_dict()}, out_path)
        print(f"saved weights: {out_path}  avg={avg:.4f}")


if __name__ == "__main__":
    main()
