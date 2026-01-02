import argparse
import glob
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from tqdm import tqdm

from models.ground_cnn import GroundModel
from models.reasoner import Reasoner
from training.geoindex.index import GeoIndex


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def latest_ckpt_path(ckpt_dir: str) -> str:
    pats = [
        os.path.join(ckpt_dir, "weights_epoch*.pth"),
        os.path.join(ckpt_dir, "*.pth"),
    ]
    cands: List[str] = []
    for p in pats:
        cands.extend(glob.glob(p))
    if not cands:
        raise FileNotFoundError(f"no .pth checkpoints found in {ckpt_dir}")

    def key_fn(p: str):
        base = os.path.basename(p)
        digits = "".join(ch for ch in base if ch.isdigit())
        if digits:
            try:
                return (0, int(digits))
            except Exception:
                pass
        try:
            return (1, int(os.path.getmtime(p)))
        except Exception:
            return (1, 0)

    cands = sorted(cands, key=key_fn)
    return cands[-1]


def haversine_km_np(lat1, lon1, lat2, lon2):
    lat1 = np.asarray(lat1, dtype=np.float64)
    lon1 = np.asarray(lon1, dtype=np.float64)
    lat2 = np.asarray(lat2, dtype=np.float64)
    lon2 = np.asarray(lon2, dtype=np.float64)
    r = 6371.0088
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    dphi = np.deg2rad(lat2 - lat1)
    dl = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * (np.sin(dl / 2.0) ** 2)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return r * c


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _img_to_tensor(path: str, img_h: int, img_w: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((img_w, img_h))
    x = np.asarray(img).astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    t = torch.from_numpy(x).permute(2, 0, 1)
    return t


def _build_eval_rows(root_processed: str, images_subdir: str, metadata_csv: str, sample_limit: int):
    images_root = Path(root_processed) / images_subdir
    meta_path = Path(root_processed) / metadata_csv
    if not images_root.exists():
        raise FileNotFoundError(f"images_root not found: {images_root}")
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata_csv not found: {meta_path}")

    id2path: Dict[str, str] = {}
    for p in images_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMG_EXTS:
            continue
        stem = p.stem
        if stem not in id2path:
            id2path[stem] = str(p)

    df = pd.read_csv(str(meta_path), low_memory=False)
    if "id" not in df.columns or "latitude" not in df.columns or "longitude" not in df.columns:
        raise ValueError(f"metadata.csv must have columns id, latitude, longitude; got {set(df.columns)}")

    rows = []
    for r in df.itertuples(index=False):
        rid = str(getattr(r, "id")).strip()
        if not rid:
            continue
        pth = id2path.get(rid, None)
        if pth is None:
            continue
        try:
            lat = float(getattr(r, "latitude"))
            lon = float(getattr(r, "longitude"))
        except Exception:
            continue
        rows.append((pth, lat, lon))

    if len(rows) == 0:
        raise RuntimeError("No eval rows matched. Check image stems vs metadata.csv id column.")

    rng = np.random.default_rng(42)
    rng.shuffle(rows)

    if sample_limit is not None and int(sample_limit) > 0:
        rows = rows[: int(sample_limit)]

    return rows


def _load_ckpt_any(path: str) -> dict:
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        return {"_raw": obj}
    return obj


def _find_first(root: Path, names: List[str]) -> Optional[Path]:
    for n in names:
        p = root / n
        if p.exists():
            return p
    return None


def _try_np_load(path: str, allow_pickle: bool, mmap_mode: Optional[str] = None):
    try:
        return np.load(path, allow_pickle=allow_pickle, mmap_mode=mmap_mode)
    except Exception:
        return None


def _extract_from_pickle_npy(x: object) -> object:
    if isinstance(x, np.ndarray) and x.dtype == object:
        if x.ndim == 0:
            return x.item()
        return x
    return x


def _load_cell_latlon_map(geo_root: str) -> Dict[int, Tuple[float, float]]:
    gr = Path(geo_root)

    pqp = gr / "cell_meta.parquet"
    if pqp.exists():
        try:
            df = pd.read_parquet(str(pqp))
            cols = set(df.columns)

            id_col = None
            for c in ["cell_id", "cell_ids", "id"]:
                if c in cols:
                    id_col = c
                    break

            lat_col = None
            for c in ["cell_lat", "lat", "latitude"]:
                if c in cols:
                    lat_col = c
                    break

            lon_col = None
            for c in ["cell_lon", "lon", "longitude"]:
                if c in cols:
                    lon_col = c
                    break

            if id_col and lat_col and lon_col:
                out = {}
                for r in df[[id_col, lat_col, lon_col]].itertuples(index=False):
                    cid = int(getattr(r, id_col))
                    out[cid] = (float(getattr(r, lat_col)), float(getattr(r, lon_col)))
                if out:
                    return out
        except Exception:
            pass

    # NPY triplets
    cand_pairs = [
        ("cell_ids.npy", "cell_lat.npy", "cell_lon.npy"),
        ("cells.npy", "cell_lat.npy", "cell_lon.npy"),
        ("cell_ids.npy", "cell_latitude.npy", "cell_longitude.npy"),
    ]
    for a, b, c in cand_pairs:
        p_ids = gr / a
        p_lat = gr / b
        p_lon = gr / c
        if p_ids.exists() and p_lat.exists() and p_lon.exists():
            ids = np.load(str(p_ids), mmap_mode="r").astype(np.int64)
            lat = np.load(str(p_lat), mmap_mode="r").astype(np.float32)
            lon = np.load(str(p_lon), mmap_mode="r").astype(np.float32)
            if ids.shape[0] != lat.shape[0] or ids.shape[0] != lon.shape[0]:
                continue
            return {int(ids[i]): (float(lat[i]), float(lon[i])) for i in range(ids.shape[0])}

    # NPZ meta
    meta_npz = gr / "cell_meta.npz"
    if meta_npz.exists():
        cm = np.load(str(meta_npz))
        if "cell_ids" in cm and "cell_lat" in cm and "cell_lon" in cm:
            ids = cm["cell_ids"].astype(np.int64)
            lat = cm["cell_lat"].astype(np.float32)
            lon = cm["cell_lon"].astype(np.float32)
            if ids.shape[0] == lat.shape[0] == lon.shape[0]:
                return {int(ids[i]): (float(lat[i]), float(lon[i])) for i in range(ids.shape[0])}

    # CSV meta
    meta_csv = gr / "cell_meta.csv"
    if meta_csv.exists():
        df = pd.read_csv(str(meta_csv))
        cols = set(df.columns)
        if "cell_id" in cols and "lat" in cols and "lon" in cols:
            return {int(r.cell_id): (float(r.lat), float(r.lon)) for r in df.itertuples(index=False)}
        if "cell_id" in cols and "cell_lat" in cols and "cell_lon" in cols:
            return {int(r.cell_id): (float(r.cell_lat), float(r.cell_lon)) for r in df.itertuples(index=False)}

    raise FileNotFoundError(
        f"Could not find cell lat/lon mapping in {geo_root}. Expected cell_meta.parquet or "
        f"cell_ids.npy + cell_lat.npy + cell_lon.npy, or cell_meta.npz."
    )


def _load_dem_cache(dem_cache_root: str, dem_h: int, dem_w: int):
    r = Path(dem_cache_root)
    if not r.exists():
        raise FileNotFoundError(f"DEM cache root not found: {dem_cache_root}")

    ids_path = _find_first(r, ["cell_ids.npy", "cells.npy", "cell_id.npy", "cell_ids.int64.npy"])
    if ids_path is None:
        raise FileNotFoundError(f"DEM cache missing cell_ids.npy in {dem_cache_root}")

    patches_path = _find_first(
        r,
        [
            "dem_patches.npy",
            "patches.npy",
            "dem.npy",
            "dem_patches.f16",
            "patches.f16",
            "dem.f16",
            "dem_patches.bin",
            "patches.bin",
            "dem.bin",
        ],
    )
    if patches_path is None:
        raise FileNotFoundError(f"DEM cache missing patches file in {dem_cache_root}")

    ids_obj = _try_np_load(str(ids_path), allow_pickle=False, mmap_mode=None)
    if ids_obj is None:
        ids_obj = _try_np_load(str(ids_path), allow_pickle=True, mmap_mode=None)
    if ids_obj is None:
        raise ValueError(f"Failed to load cell id file: {ids_path}")

    ids_obj = _extract_from_pickle_npy(ids_obj)
    if isinstance(ids_obj, dict):
        for k in ["cell_ids", "cells", "cell_id"]:
            if k in ids_obj:
                ids_obj = ids_obj[k]
                break

    ids = np.asarray(ids_obj, dtype=np.int64).reshape(-1)
    c = int(ids.shape[0])
    H = int(dem_h)
    W = int(dem_w)

    def _finalize_arr(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 4 and arr.shape[1] == 1:
            arr = arr[:, 0, :, :]
        if arr.ndim != 3:
            raise ValueError(f"DEM patches must be [C,H,W] or [C,1,H,W], got {arr.shape}")
        if int(arr.shape[0]) != c:
            raise ValueError(f"DEM patches C mismatch: ids={c} patches={arr.shape[0]}")
        if int(arr.shape[1]) != H or int(arr.shape[2]) != W:
            raise ValueError(f"DEM patch size mismatch: got={arr.shape[1:]} expected=({H},{W})")
        return arr

    if patches_path.suffix.lower() == ".npy":
        patches_obj = _try_np_load(str(patches_path), allow_pickle=False, mmap_mode="r")
        if patches_obj is None:
            patches_obj = _try_np_load(str(patches_path), allow_pickle=True, mmap_mode=None)

        if patches_obj is not None:
            patches_obj = _extract_from_pickle_npy(patches_obj)

            if isinstance(patches_obj, dict):
                for k in ["dem", "patches", "dem_patches"]:
                    if k in patches_obj:
                        patches_obj = patches_obj[k]
                        break

            if isinstance(patches_obj, np.ndarray) and patches_obj.dtype == object and patches_obj.ndim == 0:
                patches_obj = patches_obj.item()

            if isinstance(patches_obj, (list, tuple)):
                arr_list = [np.asarray(p, dtype=np.float32) for p in patches_obj]
                arr = np.stack(arr_list, axis=0)
            else:
                arr = np.asarray(patches_obj)

            try:
                patches = _finalize_arr(arr).astype(np.float32, copy=False)
            except Exception:
                # fall back to raw-blob heuristic by file size
                patches_obj = None
        else:
            patches_obj = None

        if patches_obj is None:
            fsz = os.path.getsize(str(patches_path))
            expected_elems = c * H * W
            if fsz == expected_elems * 2:
                dt = np.float16
            elif fsz == expected_elems * 4:
                dt = np.float32
            else:
                raise ValueError(
                    f"dem_patches.npy is neither readable .npy nor matching raw f16/f32 size "
                    f"(size={fsz}, expected={expected_elems*2} or {expected_elems*4}) path={patches_path}"
                )
            mm = np.memmap(str(patches_path), dtype=dt, mode="r")
            patches = mm.reshape((c, H, W))
    else:
        fsz = os.path.getsize(str(patches_path))
        expected_elems = c * H * W
        if fsz == expected_elems * 2:
            dt = np.float16
        elif fsz == expected_elems * 4:
            dt = np.float32
        else:
            dt = np.float16
        mm = np.memmap(str(patches_path), dtype=dt, mode="r")
        if int(mm.size) != int(expected_elems):
            raise ValueError(f"DEM raw size mismatch: got={mm.size} expected={expected_elems} path={patches_path}")
        patches = mm.reshape((c, H, W))

    idx = {int(ids[i]): i for i in range(c)}
    return patches, idx


@torch.no_grad()
def eval_phase2(
    rows,
    gmodel,
    reasoner,
    geoindex: GeoIndex,
    cell_map: Dict[int, Tuple[float, float]],
    dem_patches,
    dem_idx: Dict[int, int],
    img_h: int,
    img_w: int,
    dem_h: int,
    dem_w: int,
    topk: int,
    device,
):
    errs_retr = []
    errs_reasoned = []
    used_reasoner = 0

    for (pth, lat, lon) in tqdm(rows, desc="eval2", ncols=100):
        x = _img_to_tensor(pth, img_h, img_w).unsqueeze(0).to(device)
        _, _, emb = gmodel(x)
        emb = F.normalize(emb, dim=-1)

        q = emb.detach().cpu().numpy().astype(np.float32)
        _, cand_cells = geoindex.search(q, topk=int(topk))
        cand_cells = np.asarray(cand_cells, dtype=np.int64).reshape(-1)
        if cand_cells.size == 0:
            continue

        best_c = int(cand_cells[0])
        if best_c in cell_map:
            pred_lat, pred_lon = cell_map[best_c]
            e1 = float(haversine_km_np(lat, lon, pred_lat, pred_lon))
        else:
            e1 = float("inf")
        errs_retr.append(e1)

        ok = True
        dem_list = []
        for cid in cand_cells.tolist():
            j = dem_idx.get(int(cid), None)
            if j is None:
                ok = False
                break
            dem_list.append(np.asarray(dem_patches[j], dtype=np.float32))
        if not ok or len(dem_list) == 0:
            errs_reasoned.append(e1)
            continue

        dem_arr = np.stack(dem_list, axis=0) # [K,H,W]
        if dem_arr.ndim != 3:
            errs_reasoned.append(e1)
            continue
        if dem_arr.shape[1] != int(dem_h) or dem_arr.shape[2] != int(dem_w):
            errs_reasoned.append(e1)
            continue

        dem_tensor = torch.from_numpy(dem_arr).unsqueeze(0).to(device) # [1,K,H,W]

        out = reasoner(emb, dem_tensor)
        if isinstance(out, tuple):
            dem_logits = out[0]
        else:
            dem_logits = out

        if not isinstance(dem_logits, torch.Tensor):
            errs_reasoned.append(e1)
            continue
        if dem_logits.ndim != 2:
            errs_reasoned.append(e1)
            continue

        if dem_logits.shape[1] == int(cand_cells.shape[0]):
            cand_logits = dem_logits
        else:
            max_cid = int(cand_cells.max())
            if dem_logits.shape[1] <= max_cid:
                errs_reasoned.append(e1)
                continue
            idx_t = torch.from_numpy(cand_cells.reshape(1, -1)).to(device)
            cand_logits = torch.gather(dem_logits, 1, idx_t)

        j = int(torch.argmax(cand_logits, dim=1).item())
        chosen_c = int(cand_cells[j])
        if chosen_c in cell_map:
            pl, pn = cell_map[chosen_c]
            e2 = float(haversine_km_np(lat, lon, pl, pn))
        else:
            e2 = e1

        errs_reasoned.append(e2)
        used_reasoner += 1

    a1 = np.asarray(errs_retr, dtype=np.float64)
    a2 = np.asarray(errs_reasoned, dtype=np.float64)
    return a1, a2, int(used_reasoner)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-cfg", required=True)
    ap.add_argument("--grid-cfg", required=True)
    ap.add_argument("--model-cfg", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--geo-root", default="")
    ap.add_argument("--dem-cache", default="")
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--sample-limit", type=int, default=1000)
    args = ap.parse_args()

    data_cfg = load_yaml(args.data_cfg)
    grid_cfg = load_yaml(args.grid_cfg)
    model_cfg = load_yaml(args.model_cfg)

    root_processed = str(data_cfg.get("processed_root"))
    images_subdir = str(data_cfg.get("images_subdir", "images"))
    metadata_csv = str(data_cfg.get("metadata_csv", "metadata.csv"))

    img_h = int(model_cfg.get("img_h", 224))
    img_w = int(model_cfg.get("img_w", 224))
    dem_h = int(model_cfg.get("dem_h", 64))
    dem_w = int(model_cfg.get("dem_w", 64))
    embed_dim = int(model_cfg.get("embed_dim", 256))

    lat_bins = int(grid_cfg.get("lat_bins", 0) or 0)
    lon_bins = int(grid_cfg.get("lon_bins", 0) or 0)
    num_cells = int(lat_bins * lon_bins) if lat_bins > 0 and lon_bins > 0 else 1

    if not args.geo_root:
        args.geo_root = str((data_cfg.get("geoindex", {}) or {}).get("root", "/mnt/data/geoindex"))
    geo_root = args.geo_root

    if not args.dem_cache:
        args.dem_cache = str((data_cfg.get("dem_cache", {}) or {}).get("root", os.path.join(geo_root, "dem_cache")))
    dem_cache_root = args.dem_cache

    rows = _build_eval_rows(
        root_processed=root_processed,
        images_subdir=images_subdir,
        metadata_csv=metadata_csv,
        sample_limit=int(args.sample_limit),
    )

    device = _device()

    gmodel = GroundModel.from_config(
        {
            "num_cells": num_cells,
            "embed_dim": embed_dim,
            "backbone": model_cfg.get("backbone", "resnet18"),
            "pretrained_backbone": bool(model_cfg.get("pretrained_backbone", True)),
            "in_chans": 3,
            "dropout": float(model_cfg.get("dropout", 0.1)),
        }
    ).to(device).eval()

    reason_hidden = int(model_cfg.get("reason_hidden", 256))
    dem_embed_dim = int(model_cfg.get("dem_embed_dim", 128))
    reasoner = Reasoner(
        img_embed_dim=embed_dim,
        num_cells=num_cells,
        dem_embed_dim=dem_embed_dim,
        hidden=reason_hidden,
        k=int(args.topk),
    ).to(device).eval()

    ckpt_path = latest_ckpt_path(args.ckpt_dir)
    print(f"[eval2] using checkpoint: {ckpt_path}")
    ckpt = _load_ckpt_any(ckpt_path)

    # load gmodel
    if "gmodel" in ckpt and isinstance(ckpt["gmodel"], dict):
        gmodel.load_state_dict(ckpt["gmodel"], strict=False)
    elif "model" in ckpt and isinstance(ckpt["model"], dict):
        gmodel.load_state_dict(ckpt["model"], strict=False)
    else:
        try:
            gmodel.load_state_dict(ckpt, strict=False)
        except Exception:
            pass

    # load reasoner
    if "reasoner" in ckpt and isinstance(ckpt["reasoner"], dict):
        reasoner.load_state_dict(ckpt["reasoner"], strict=False)

    geoindex = GeoIndex(root=str(geo_root))
    cell_map = _load_cell_latlon_map(str(geo_root))
    dem_patches, dem_idx = _load_dem_cache(str(dem_cache_root), dem_h=dem_h, dem_w=dem_w)

    errs_retr, errs_reasoned, used = eval_phase2(
        rows=rows,
        gmodel=gmodel,
        reasoner=reasoner,
        geoindex=geoindex,
        cell_map=cell_map,
        dem_patches=dem_patches,
        dem_idx=dem_idx,
        img_h=img_h,
        img_w=img_w,
        dem_h=dem_h,
        dem_w=dem_w,
        topk=int(args.topk),
        device=device,
    )

    m1 = float(np.mean(errs_retr)) if errs_retr.size else float("inf")
    d1 = float(np.median(errs_retr)) if errs_retr.size else float("inf")
    m2 = float(np.mean(errs_reasoned)) if errs_reasoned.size else float("inf")
    d2 = float(np.median(errs_reasoned)) if errs_reasoned.size else float("inf")

    r25 = float(np.mean(errs_reasoned <= 25.0)) if errs_reasoned.size else 0.0
    r100 = float(np.mean(errs_reasoned <= 100.0)) if errs_reasoned.size else 0.0
    r500 = float(np.mean(errs_reasoned <= 500.0)) if errs_reasoned.size else 0.0

    print(f"[eval2] retrieval top1: mean={m1:.2f} km | median={d1:.2f} km")
    print(f"[eval2] reasoned: mean={m2:.2f} km | median={d2:.2f} km | used_reasoner={used}/{len(rows)}")
    print(f"[eval2] reasoned recall: <=25km {r25:.3f} | <=100km {r100:.3f} | <=500km {r500:.3f}")


if __name__ == "__main__":
    main()
