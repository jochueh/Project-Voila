import argparse
import sys
import yaml
import csv
import math
import os
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
from numpy.lib.format import open_memmap
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms as T


def load_meta(csv_path: str, id_col: str, lat_col: str, lon_col: str):
    m = {}
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                k = (row[id_col] or "").strip()
                lat = float(row[lat_col])
                lon = float(row[lon_col])
            except Exception:
                continue
            if k:
                m[k] = (lat, lon)
    return m


def geo_cell_id(lat: float, lon: float, cell_size_deg: float) -> int:
    lat_clamped = max(min(lat, 89.999999), -89.999999)
    lon_wrapped = ((lon + 180.0) % 360.0) - 180.0

    n_lat = int(round(180.0 / cell_size_deg))
    n_lon = int(round(360.0 / cell_size_deg))

    i_lat = int(math.floor((lat_clamped + 90.0) / cell_size_deg))
    i_lon = int(math.floor((lon_wrapped + 180.0) / cell_size_deg))

    i_lat = max(0, min(n_lat - 1, i_lat))
    i_lon = max(0, min(n_lon - 1, i_lon))

    return i_lat * n_lon + i_lon


def _resolve_ckpt(path: str):
    if not path:
        return None
    pth = Path(path)
    if pth.is_dir():
        cand = sorted(pth.glob("*.pth"), key=lambda x: x.stat().st_mtime)
        if not cand:
            raise FileNotFoundError(f"No .pth in {pth}")
        return str(cand[-1])
    return str(pth)


def _safe_unlink(p: Path):
    try:
        if p.exists():
            p.unlink()
    except Exception:
        pass


def _safe_rmtree(p: Path):
    try:
        if p.is_dir():
            for child in p.glob("*"):
                if child.is_dir():
                    _safe_rmtree(child)
                else:
                    _safe_unlink(child)
            p.rmdir()
    except Exception:
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--images_root", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--grid_cfg", required=True)
    p.add_argument("--model_cfg", required=True)
    p.add_argument("--out_root", required=True)
    p.add_argument("--ckpt", default=None)

    p.add_argument("--id_col", default="id")
    p.add_argument("--lat_col", default="latitude")
    p.add_argument("--lon_col", default="longitude")

    p.add_argument("--limit", type=int, default=0, help="0 = no limit")
    p.add_argument("--log_every", type=int, default=2000)

    p.add_argument("--overwrite", action="store_true", help="delete existing outputs in out_root before writing")
    p.add_argument("--no_sort", action="store_true", help="do not sort by geo cell id (faster + less extra disk)")
    p.add_argument("--sort_chunk", type=int, default=200_000, help="chunk size when sorting/reordering")

    args = p.parse_args()

    sys.path.append(args.repo)
    from models.ground_cnn import GroundModel

    with open(args.grid_cfg) as f:
        grid = yaml.safe_load(f)
    with open(args.model_cfg) as f:
        model_cfg = yaml.safe_load(f)

    lat_bins = int(grid["lat_bins"])
    lon_bins = int(grid["lon_bins"])
    img_h = int(model_cfg.get("img_h", 224))
    img_w = int(model_cfg.get("img_w", 224))
    embed_dim = int(model_cfg.get("embed_dim", 256))

    geo_cfg = grid.get("geoindex", {}) or {}
    cell_size_deg = float(geo_cfg.get("cell_size_deg", 5.0))

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    embeddings_path = out_root / "embeddings.npy"
    cid_path = out_root / "cell_id.npy"
    lat_path = out_root / "lat.npy"
    lon_path = out_root / "lon.npy"
    sample_meta_npz = out_root / "sample_meta.npz"
    cell_meta_npz = out_root / "cell_meta.npz"

    tmp_embeddings_path = out_root / "embeddings_unsorted.npy"
    tmp_cid_path = out_root / "cell_id_unsorted.npy"
    tmp_lat_path = out_root / "lat_unsorted.npy"
    tmp_lon_path = out_root / "lon_unsorted.npy"

    if args.overwrite:
        _safe_unlink(embeddings_path)
        _safe_unlink(cid_path)
        _safe_unlink(lat_path)
        _safe_unlink(lon_path)
        _safe_unlink(sample_meta_npz)
        _safe_unlink(cell_meta_npz)
        _safe_unlink(tmp_embeddings_path)
        _safe_unlink(tmp_cid_path)
        _safe_unlink(tmp_lat_path)
        _safe_unlink(tmp_lon_path)

    meta = load_meta(args.meta, args.id_col, args.lat_col, args.lon_col)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gmodel = GroundModel.from_config({
        "num_cells": lat_bins * lon_bins,
        "embed_dim": embed_dim,
        "backbone": model_cfg.get("backbone", "resnet18"),
        "pretrained_backbone": model_cfg.get("pretrained_backbone", True),
        "in_chans": 3,
        "dropout": model_cfg.get("dropout", 0.1),
    }).to(device).eval()

    ckpt = _resolve_ckpt(args.ckpt)
    if ckpt is not None:
        state = torch.load(ckpt, map_location=device)
        if isinstance(state, dict) and "model" in state:
            gmodel.load_state_dict(state["model"], strict=False)
        else:
            gmodel.load_state_dict(state, strict=False)

    tf = T.Compose([
        T.Resize((img_h, img_w), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    images_root = Path(args.images_root)
    limit = int(args.limit) if args.limit and args.limit > 0 else 0

    t0 = time.time()
    total_matches = 0
    total_scanned = 0

    for fp in images_root.rglob("*"):
        if not fp.is_file():
            continue
        if fp.suffix.lower() not in exts:
            continue
        total_scanned += 1
        key = fp.stem
        if key in meta:
            total_matches += 1
            if limit and total_matches >= limit:
                break

    if total_matches == 0:
        raise SystemExit("No matches found (no image stems matched metadata ids).")

    if args.no_sort:
        X_path = embeddings_path
        C_path = cid_path
        La_path = lat_path
        Lo_path = lon_path
    else:
        X_path = tmp_embeddings_path
        C_path = tmp_cid_path
        La_path = tmp_lat_path
        Lo_path = tmp_lon_path

    X = open_memmap(str(X_path), mode="w+", dtype="float32", shape=(total_matches, embed_dim))
    C = open_memmap(str(C_path), mode="w+", dtype="int64", shape=(total_matches,))
    La = open_memmap(str(La_path), mode="w+", dtype="float32", shape=(total_matches,))
    Lo = open_memmap(str(Lo_path), mode="w+", dtype="float32", shape=(total_matches,))

    cell_lat_sum = defaultdict(float)
    cell_lon_sum = defaultdict(float)
    cell_count = defaultdict(int)

    written = 0
    scanned = 0
    last_log = time.time()

    def load_img_tensor(path: Path):
        img = Image.open(path).convert("RGB")
        return tf(img).unsqueeze(0)

    with torch.no_grad():
        for fp in images_root.rglob("*"):
            if not fp.is_file():
                continue
            if fp.suffix.lower() not in exts:
                continue

            scanned += 1
            key = fp.stem
            info = meta.get(key)
            if info is None:
                continue

            lat, lon = info
            x = load_img_tensor(fp).to(device)

            _, _, emb = gmodel(x)
            e = F.normalize(emb, dim=-1).squeeze(0).cpu().numpy().astype("float32")

            cid_geo = geo_cell_id(lat, lon, cell_size_deg)

            X[written, :] = e
            C[written] = int(cid_geo)
            La[written] = float(lat)
            Lo[written] = float(lon)

            cell_lat_sum[int(cid_geo)] += float(lat)
            cell_lon_sum[int(cid_geo)] += float(lon)
            cell_count[int(cid_geo)] += 1

            written += 1

            if written >= total_matches:
                break

            if args.log_every > 0 and written % args.log_every == 0:
                now = time.time()
                dt = now - last_log
                total_dt = now - t0
                ips = args.log_every / max(dt, 1e-6)
                avg_ips = written / max(total_dt, 1e-6)
                print(f"[geoindex] {written}/{total_matches} written | scanned {scanned} | {ips:.1f} img/s (recent) | {avg_ips:.1f} img/s (avg)")
                X.flush()
                C.flush()
                La.flush()
                Lo.flush()
                last_log = now

    X.flush()
    C.flush()
    La.flush()
    Lo.flush()

    if written != total_matches:
        total_matches = written
        X = np.load(str(X_path), mmap_mode="r+")
        C = np.load(str(C_path), mmap_mode="r+")
        La = np.load(str(La_path), mmap_mode="r+")
        Lo = np.load(str(Lo_path), mmap_mode="r+")

    if not args.no_sort:
        C_in = np.load(str(tmp_cid_path), mmap_mode="r")
        order = np.argsort(C_in)

        X_out = open_memmap(str(embeddings_path), mode="w+", dtype="float32", shape=(total_matches, embed_dim))
        C_out = open_memmap(str(cid_path), mode="w+", dtype="int64", shape=(total_matches,))
        La_out = open_memmap(str(lat_path), mode="w+", dtype="float32", shape=(total_matches,))
        Lo_out = open_memmap(str(lon_path), mode="w+", dtype="float32", shape=(total_matches,))

        X_in = np.load(str(tmp_embeddings_path), mmap_mode="r")
        La_in = np.load(str(tmp_lat_path), mmap_mode="r")
        Lo_in = np.load(str(tmp_lon_path), mmap_mode="r")

        chunk = max(10_000, int(args.sort_chunk))
        for i in range(0, total_matches, chunk):
            j = min(total_matches, i + chunk)
            idx = order[i:j]
            X_out[i:j, :] = X_in[idx, :]
            C_out[i:j] = C_in[idx]
            La_out[i:j] = La_in[idx]
            Lo_out[i:j] = Lo_in[idx]
            if (i // chunk) % 5 == 0:
                X_out.flush()
                C_out.flush()
                La_out.flush()
                Lo_out.flush()

        X_out.flush()
        C_out.flush()
        La_out.flush()
        Lo_out.flush()

        _safe_unlink(tmp_embeddings_path)
        _safe_unlink(tmp_cid_path)
        _safe_unlink(tmp_lat_path)
        _safe_unlink(tmp_lon_path)

        X = np.load(str(embeddings_path), mmap_mode="r")
        C = np.load(str(cid_path), mmap_mode="r")
        La = np.load(str(lat_path), mmap_mode="r")
        Lo = np.load(str(lon_path), mmap_mode="r")

    C_final = np.load(str(cid_path), mmap_mode="r")
    cell_ids, counts = np.unique(C_final, return_counts=True)
    offsets = np.cumsum(counts)
    starts = offsets - counts
    ends = offsets

    cell_lat = np.empty_like(cell_ids, dtype="float32")
    cell_lon = np.empty_like(cell_ids, dtype="float32")
    for i, cid in enumerate(cell_ids):
        cnt = cell_count.get(int(cid), 0)
        if cnt > 0:
            cell_lat[i] = float(cell_lat_sum[int(cid)] / cnt)
            cell_lon[i] = float(cell_lon_sum[int(cid)] / cnt)
        else:
            cell_lat[i] = 0.0
            cell_lon[i] = 0.0

    np.savez(sample_meta_npz, cell_id=np.asarray(C_final, dtype="int64"),
             lat=np.asarray(np.load(str(lat_path), mmap_mode="r"), dtype="float32"),
             lon=np.asarray(np.load(str(lon_path), mmap_mode="r"), dtype="float32"))

    np.savez(cell_meta_npz,
             cell_ids=cell_ids.astype("int64"),
             start=starts.astype("int64"),
             end=ends.astype("int64"),
             cell_lat=cell_lat.astype("float32"),
             cell_lon=cell_lon.astype("float32"),
             cell_size_deg=np.array([cell_size_deg], dtype="float32"))

    t1 = time.time()
    print(f"[geoindex] done: {str(out_root)} | N={total_matches} | D={embed_dim} | cells={cell_ids.shape[0]} | {total_matches/max(t1-t0,1e-6):.2f} img/s")


if __name__ == "__main__":
    main()
