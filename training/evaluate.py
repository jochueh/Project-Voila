import argparse
import os
import glob
import yaml
import numpy as np
import torch

from training.datasets import EvalGroundDataset
from training.losses import haversine_km
from models.ground_cnn import GroundModel


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def latest_ckpt(ckpt_dir: str) -> str:
    cand = glob.glob(os.path.join(ckpt_dir, "weights_epoch*.pth"))
    if not cand:
        cand = glob.glob(os.path.join(ckpt_dir, "*.pth"))
    if not cand:
        raise FileNotFoundError(f"No .pth checkpoints in {ckpt_dir}")
    cand.sort(key=lambda p: os.path.getmtime(p))
    return cand[-1]


def load_model_from_ckpt(gmodel: torch.nn.Module, ckpt_path: str, device: str):
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        gmodel.load_state_dict(state["model"], strict=False)
    else:
        gmodel.load_state_dict(state, strict=False)
    return gmodel


@torch.no_grad()
def eval_one_ckpt(gmodel: torch.nn.Module, ds: EvalGroundDataset, device: str):
    gmodel.eval()
    errs = []
    for i in range(len(ds)):
        img, lat, lon = ds[i]
        x = img.unsqueeze(0).to(device)
        logits, offsets, _ = gmodel(x)

        cid = int(torch.argmax(logits, dim=1).item())
        off = offsets.squeeze(0).detach().cpu().numpy().astype(np.float32)

        pred_lat, pred_lon = ds.decode_cell_and_offset(cid, off)

        lat1 = torch.tensor(lat, dtype=torch.float32)
        lon1 = torch.tensor(lon, dtype=torch.float32)
        lat2 = torch.tensor(pred_lat, dtype=torch.float32)
        lon2 = torch.tensor(pred_lon, dtype=torch.float32)
        errs.append(float(haversine_km(lat1, lon1, lat2, lon2).item()))

    errs = np.asarray(errs, dtype=np.float32)
    mean_km = float(errs.mean()) if errs.size else float("nan")
    med_km = float(np.median(errs)) if errs.size else float("nan")
    return mean_km, med_km


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-cfg", required=True)
    p.add_argument("--grid-cfg", required=True)
    p.add_argument("--model-cfg", required=True)
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--sample-limit", type=int, default=1000)
    p.add_argument("--all", action="store_true", help="evaluate all weights_epoch*.pth in ckpt-dir")
    args = p.parse_args()

    data_cfg = load_yaml(args.data_cfg)
    grid_cfg = load_yaml(args.grid_cfg)
    model_cfg = load_yaml(args.model_cfg)

    lat_bins = int(grid_cfg["lat_bins"])
    lon_bins = int(grid_cfg["lon_bins"])
    img_h = int(model_cfg.get("img_h", 224))
    img_w = int(model_cfg.get("img_w", 224))
    embed_dim = int(model_cfg.get("embed_dim", 256))

    root_processed = str(data_cfg.get("processed_root") or data_cfg.get("root"))
    images_subdir = str(data_cfg.get("images_subdir", "images"))
    metadata_csv = str(data_cfg.get("metadata_csv", "metadata.csv"))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = EvalGroundDataset(
        root_processed=root_processed,
        img_size_hw=(img_h, img_w),
        sample_limit=args.sample_limit,
        images_subdir=images_subdir,
        metadata_csv=metadata_csv,
        lat_bins=lat_bins,
        lon_bins=lon_bins,
        id_col=str(data_cfg.get("id_col", "id")),
        lat_col=str(data_cfg.get("lat_col", "latitude")),
        lon_col=str(data_cfg.get("lon_col", "longitude")),
    )

    if len(ds) == 0:
        raise SystemExit("EvalGroundDataset is empty (no images matched metadata ids).")

    gmodel = GroundModel.from_config({
        "num_cells": lat_bins * lon_bins,
        "embed_dim": embed_dim,
        "backbone": model_cfg.get("backbone", "resnet18"),
        "pretrained_backbone": model_cfg.get("pretrained_backbone", True),
        "in_chans": 3,
        "dropout": model_cfg.get("dropout", 0.1),
    }).to(device)

    if args.all:
        ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, "weights_epoch*.pth")))
        if not ckpts:
            raise FileNotFoundError(f"No weights_epoch*.pth in {args.ckpt_dir}")
        for ck in ckpts:
            load_model_from_ckpt(gmodel, ck, device)
            mean_km, med_km = eval_one_ckpt(gmodel, ds, device)
            print(f"{os.path.basename(ck)} -> Mean {mean_km:.2f} km, Median {med_km:.2f} km")
    else:
        ck = latest_ckpt(args.ckpt_dir)
        load_model_from_ckpt(gmodel, ck, device)
        mean_km, med_km = eval_one_ckpt(gmodel, ds, device)
        print(f"{os.path.basename(ck)} -> Mean {mean_km:.2f} km, Median {med_km:.2f} km")


if __name__ == "__main__":
    main()
