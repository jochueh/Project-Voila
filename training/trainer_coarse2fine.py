import argparse
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from models.ground_cnn import GroundModel
from models.reasoner import Reasoner
from training.geoindex.index import GeoIndex

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed_all(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _make_run_dir(base_dir: str) -> str:
    os.makedirs(base_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run = os.path.join(base_dir, stamp)
    os.makedirs(run, exist_ok=True)
    return run


def _geo_cell_id(lat: float, lon: float, cell_size_deg: float) -> int:
    lat_clamped = max(min(float(lat), 89.999999), -89.999999)
    lon_wrapped = ((float(lon) + 180.0) % 360.0) - 180.0
    n_lat = int(round(180.0 / cell_size_deg))
    n_lon = int(round(360.0 / cell_size_deg))
    i_lat = int(math.floor((lat_clamped + 90.0) / cell_size_deg))
    i_lon = int(math.floor((lon_wrapped + 180.0) / cell_size_deg))
    i_lat = max(0, min(n_lat - 1, i_lat))
    i_lon = max(0, min(n_lon - 1, i_lon))
    return i_lat * n_lon + i_lon


def _find_first(root: Path, names: List[str]) -> Optional[Path]:
    for n in names:
        p = root / n
        if p.exists():
            return p
    return None


def _load_state_dict_like(path: str) -> Dict:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        return obj["model"]
    if isinstance(obj, dict) and "gmodel" in obj and isinstance(obj["gmodel"], dict):
        return obj["gmodel"]
    if isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
        return obj
    raise ValueError(f"Unrecognized checkpoint format: {path}")


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


class DemCellCache:
    def __init__(self, cache_root: str, dem_h: int, dem_w: int):
        r = Path(cache_root)
        if not r.exists():
            raise FileNotFoundError(f"DEM cache root not found: {cache_root}")

        ids_path = _find_first(r, ["cell_ids.npy", "cells.npy", "cell_id.npy", "cell_ids.int64.npy"])
        if ids_path is None:
            raise FileNotFoundError(f"DEM cache missing cell_ids.npy in {cache_root}")

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
            raise FileNotFoundError(f"DEM cache missing patches array in {cache_root}")

        self.dem_h = int(dem_h)
        self.dem_w = int(dem_w)

        ids_obj = _try_np_load(str(ids_path), allow_pickle=False, mmap_mode=None)
        if ids_obj is None:
            ids_obj = _try_np_load(str(ids_path), allow_pickle=True, mmap_mode=None)
        if ids_obj is None:
            raise ValueError(f"Failed to load cell id file: {ids_path}")

        ids_obj = _extract_from_pickle_npy(ids_obj)

        if isinstance(ids_obj, dict):
            if "cell_ids" in ids_obj:
                ids_obj = ids_obj["cell_ids"]
            elif "cells" in ids_obj:
                ids_obj = ids_obj["cells"]
            elif "cell_id" in ids_obj:
                ids_obj = ids_obj["cell_id"]

        self.cell_ids = np.asarray(ids_obj, dtype=np.int64).reshape(-1)
        c = int(self.cell_ids.shape[0])

        if patches_path.suffix.lower() == ".npy":
            patches_obj = _try_np_load(str(patches_path), allow_pickle=False, mmap_mode="r")
            if patches_obj is None:
                patches_obj = _try_np_load(str(patches_path), allow_pickle=True, mmap_mode=None)

            if patches_obj is not None:
                patches_obj = _extract_from_pickle_npy(patches_obj)

                if isinstance(patches_obj, dict):
                    if "dem" in patches_obj:
                        patches_obj = patches_obj["dem"]
                    elif "patches" in patches_obj:
                        patches_obj = patches_obj["patches"]
                    elif "dem_patches" in patches_obj:
                        patches_obj = patches_obj["dem_patches"]

                if isinstance(patches_obj, np.ndarray) and patches_obj.dtype == object:
                    if patches_obj.ndim == 0:
                        patches_obj = patches_obj.item()

                if isinstance(patches_obj, (list, tuple)):
                    arr_list = [np.asarray(p, dtype=np.float32) for p in patches_obj]
                    arr = np.stack(arr_list, axis=0)
                else:
                    arr = np.asarray(patches_obj)

                if arr.ndim == 4 and arr.shape[1] == 1:
                    arr = arr[:, 0, :, :]

                if arr.ndim != 3:
                    raise ValueError(f"DEM patches must be 3D [C,H,W], got {arr.shape}")

                if int(arr.shape[0]) != c:
                    raise ValueError(f"DEM patches C mismatch: ids={c} patches={arr.shape[0]}")

                if int(arr.shape[1]) != self.dem_h or int(arr.shape[2]) != self.dem_w:
                    raise ValueError(
                        f"DEM patch size mismatch: got={arr.shape[1:]} expected=({self.dem_h},{self.dem_w})"
                    )

                self.patches = arr.astype(np.float32, copy=False)
            else:
                fsz = os.path.getsize(str(patches_path))
                expected_elems = c * self.dem_h * self.dem_w
                if fsz == expected_elems * 2:
                    dt = np.float16
                elif fsz == expected_elems * 4:
                    dt = np.float32
                else:
                    raise ValueError(
                        f"dem_patches.npy is not a valid .npy and size does not match raw f16/f32 "
                        f"(size={fsz}, expected={expected_elems*2} or {expected_elems*4})"
                    )
                mm = np.memmap(str(patches_path), dtype=dt, mode="r")
                self.patches = mm.reshape((c, self.dem_h, self.dem_w))
        else:
            fsz = os.path.getsize(str(patches_path))
            expected_elems = c * self.dem_h * self.dem_w
            if fsz == expected_elems * 2:
                dt = np.float16
            elif fsz == expected_elems * 4:
                dt = np.float32
            else:
                dt = np.float16
            mm = np.memmap(str(patches_path), dtype=dt, mode="r")
            if int(mm.size) != int(expected_elems):
                raise ValueError(f"DEM raw size mismatch: got={mm.size} expected={expected_elems} path={patches_path}")
            self.patches = mm.reshape((c, self.dem_h, self.dem_w))

        self._id_to_row = {int(cid): i for i, cid in enumerate(self.cell_ids.tolist())}

    def get(self, cell_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(cell_ids, dtype=np.int64)
        if ids.ndim == 1:
            ids = ids[None, :]
        if ids.ndim != 2:
            raise ValueError(f"cell_ids must be [B,K] or [K], got {ids.shape}")

        b, k = int(ids.shape[0]), int(ids.shape[1])
        out = np.zeros((b, k, self.dem_h, self.dem_w), dtype=np.float32)

        for i in range(b):
            for j in range(k):
                cid = int(ids[i, j])
                row = self._id_to_row.get(cid, None)
                if row is None:
                    continue
                patch = self.patches[row]
                out[i, j] = np.asarray(patch, dtype=np.float32)

        return out


class GroundLatLonDataset(Dataset):
    def __init__(
        self,
        root_processed: str,
        images_subdir: str,
        metadata_csv: str,
        img_size_hw: Tuple[int, int],
        sample_limit: Optional[int] = None,
        shuffle: bool = True,
    ):
        self.root_processed = str(root_processed)
        self.images_root = str(Path(root_processed) / images_subdir)
        self.meta_path = str(Path(root_processed) / metadata_csv)
        self.img_h, self.img_w = int(img_size_hw[0]), int(img_size_hw[1])

        img_root = Path(self.images_root)
        if not img_root.exists():
            raise FileNotFoundError(f"images_root not found: {self.images_root}")

        id2path: Dict[str, str] = {}
        for p in img_root.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in IMG_EXTS:
                continue
            stem = p.stem
            if stem not in id2path:
                id2path[stem] = str(p)

        import pandas as pd

        df = pd.read_csv(self.meta_path, low_memory=False)
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

        if shuffle:
            rng = np.random.default_rng(42)
            rng.shuffle(rows)

        if sample_limit is not None and int(sample_limit) > 0:
            rows = rows[: int(sample_limit)]

        if len(rows) == 0:
            raise RuntimeError("No (image,id,lat,lon) matches found. Check ids vs image stems.")

        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def _img_to_tensor(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB").resize((self.img_w, self.img_h))
        x = np.asarray(img).astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        t = torch.from_numpy(x).permute(2, 0, 1)
        return t

    def __getitem__(self, idx: int):
        pth, lat, lon = self.rows[idx]
        return {"image": self._img_to_tensor(pth), "lat": float(lat), "lon": float(lon)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-cfg", default="configs/data.yaml")
    parser.add_argument("--grid-cfg", default="configs/grid.yaml")
    parser.add_argument("--model-cfg", default="configs/model.yaml")
    parser.add_argument("--train-cfg", default="configs/train_phase2.yaml")
    parser.add_argument("--runs-dir", default="checkpoints")
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--ckpt", default="")
    args = parser.parse_args()

    data_cfg = _load_yaml(args.data_cfg)
    grid_cfg = _load_yaml(args.grid_cfg)
    model_cfg = _load_yaml(args.model_cfg)
    train_cfg = _load_yaml(args.train_cfg)

    seed = int(train_cfg.get("seed", 42))
    _seed_all(seed)
    device = _device()

    root_processed = str(data_cfg.get("processed_root"))
    images_subdir = str(data_cfg.get("images_subdir", "images"))
    metadata_csv = str(data_cfg.get("metadata_csv", "metadata.csv"))

    img_h = int(model_cfg.get("img_h", 224))
    img_w = int(model_cfg.get("img_w", 224))
    dem_h = int(model_cfg.get("dem_h", 64))
    dem_w = int(model_cfg.get("dem_w", 64))
    embed_dim = int(model_cfg.get("embed_dim", 256))

    batch_size = int(train_cfg.get("batch_size", 64))
    workers = int(train_cfg.get("workers", 6))
    epochs = int(train_cfg.get("epochs", 10))
    log_every = int(train_cfg.get("log_every", 100))
    sample_limit = train_cfg.get("sample_limit", None)
    freeze_gmodel = bool(train_cfg.get("freeze_gmodel", True))

    geo_cfg = data_cfg.get("geoindex", {}) or {}
    geo_root = str(geo_cfg.get("root", "/mnt/data/geoindex"))
    cell_size_deg = float((grid_cfg.get("geoindex", {}) or {}).get("cell_size_deg", 5.0))
    dem_cache_root = str((data_cfg.get("dem_cache", {}) or {}).get("root", os.path.join(geo_root, "dem_cache")))

    lat_bins = int(grid_cfg.get("lat_bins", 0) or 0)
    lon_bins = int(grid_cfg.get("lon_bins", 0) or 0)
    num_cells = int(lat_bins * lon_bins) if lat_bins > 0 and lon_bins > 0 else 1

    ds = GroundLatLonDataset(
        root_processed=root_processed,
        images_subdir=images_subdir,
        metadata_csv=metadata_csv,
        img_size_hw=(img_h, img_w),
        sample_limit=sample_limit if sample_limit is None else int(sample_limit),
        shuffle=True,
    )
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
    )

    gmodel = GroundModel.from_config(
        {
            "num_cells": num_cells,
            "embed_dim": embed_dim,
            "backbone": model_cfg.get("backbone", "resnet18"),
            "pretrained_backbone": model_cfg.get("pretrained_backbone", True),
            "in_chans": 3,
            "dropout": model_cfg.get("dropout", 0.1),
        }
    ).to(device)

    reason_hidden = int(model_cfg.get("reason_hidden", 256))
    dem_embed_dim = int(model_cfg.get("dem_embed_dim", 128))
    topk = int(args.topk)

    reasoner = Reasoner(
        img_embed_dim=embed_dim,
        num_cells=num_cells,
        dem_embed_dim=dem_embed_dim,
        hidden=reason_hidden,
        k=topk,
    ).to(device)

    if args.ckpt:
        sd = _load_state_dict_like(args.ckpt)
        missing, unexpected = gmodel.load_state_dict(sd, strict=False)
        print(f"[phase2] loaded ckpt into gmodel: {args.ckpt} | missing={len(missing)} unexpected={len(unexpected)}")

    if freeze_gmodel:
        gmodel.eval()
        for p in gmodel.parameters():
            p.requires_grad = False
    else:
        gmodel.train()

    coarse = GeoIndex(root=geo_root)
    dem_cache = DemCellCache(cache_root=dem_cache_root, dem_h=dem_h, dem_w=dem_w)

    params = list(reasoner.parameters())
    if not freeze_gmodel:
        params += list(gmodel.parameters())

    lr = float(train_cfg.get("lr", 1e-4))
    wd = float(train_cfg.get("weight_decay", 0.05))
    optim = AdamW(params, lr=lr, weight_decay=wd)

    steps_per_epoch = max(1, int(math.ceil(len(dl))))
    tmax = int(epochs * steps_per_epoch)
    sched = CosineAnnealingLR(optim, T_max=tmax)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    run_dir = _make_run_dir(args.runs_dir)
    for name, cfg in [("data.yaml", data_cfg), ("grid.yaml", grid_cfg), ("model.yaml", model_cfg), ("train.yaml", train_cfg)]:
        with open(os.path.join(run_dir, name), "w") as f:
            yaml.safe_dump(cfg, f)

    for ep in range(1, epochs + 1):
        if freeze_gmodel:
            gmodel.eval()
        else:
            gmodel.train()
        reasoner.train()

        total = 0.0
        total_rank = 0.0
        total_kept = 0

        for it, batch in enumerate(dl, 1):
            img = batch["image"].to(device, non_blocking=True)
            lat = torch.as_tensor(batch["lat"], device=device, dtype=torch.float32)
            lon = torch.as_tensor(batch["lon"], device=device, dtype=torch.float32)

            optim.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                if freeze_gmodel:
                    with torch.no_grad():
                        _, _, embeds = gmodel(img)
                else:
                    _, _, embeds = gmodel(img)

            embeds_norm = F.normalize(embeds.float(), dim=-1)
            q = embeds_norm.detach().cpu().numpy().astype(np.float32)
            _, cand_cells = coarse.search(q, topk=topk)

            gt_cells = torch.tensor(
                [_geo_cell_id(float(lat[i].item()), float(lon[i].item()), cell_size_deg) for i in range(lat.shape[0])],
                dtype=torch.long,
                device=device,
            )
            cand_cells_t = torch.from_numpy(cand_cells.astype(np.int64)).to(device)

            target_idx = torch.full((cand_cells_t.shape[0],), -1, dtype=torch.long, device=device)
            for b in range(cand_cells_t.shape[0]):
                where = (cand_cells_t[b] == gt_cells[b]).nonzero(as_tuple=False)
                if where.numel() > 0:
                    target_idx[b] = int(where[0].item())

            keep = target_idx >= 0
            kept_n = int(keep.sum().item())
            if kept_n == 0:
                continue

            cand_cells_kept = cand_cells_t[keep]
            embeds_kept = embeds_norm[keep]
            target_idx_kept = target_idx[keep]

            dem_np = dem_cache.get(cand_cells_kept.detach().cpu().numpy())
            dem = torch.from_numpy(dem_np).to(device)

            if dem.ndim != 4:
                raise ValueError(f"DEM tensor must be [B,K,H,W], got {tuple(dem.shape)}")

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out = reasoner(embeds_kept, dem)
                if isinstance(out, tuple):
                    dem_logits = out[0]
                else:
                    dem_logits = out

                if dem_logits.ndim != 2:
                    raise ValueError(f"Reasoner logits must be 2D, got {tuple(dem_logits.shape)}")

                if dem_logits.shape[1] == cand_cells_kept.shape[1]:
                    cand_logits = dem_logits
                else:
                    if dem_logits.shape[1] <= int(cand_cells_kept.max().item()):
                        raise ValueError(
                            f"Reasoner logits dim1={dem_logits.shape[1]} cannot index max_cell={int(cand_cells_kept.max().item())}"
                        )
                    cand_logits = torch.gather(dem_logits, 1, cand_cells_kept)

                if cand_logits.shape != cand_cells_kept.shape:
                    raise ValueError(
                        f"Candidate logits must be [B,K]={tuple(cand_cells_kept.shape)}, got {tuple(cand_logits.shape)}"
                    )

                loss_rank = F.cross_entropy(cand_logits, target_idx_kept)
                loss = loss_rank

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            sched.step()

            total += float(loss.item()) * kept_n
            total_rank += float(loss_rank.item()) * kept_n
            total_kept += kept_n

            if log_every > 0 and it % log_every == 0:
                avg = total / max(1, total_kept)
                avgr = total_rank / max(1, total_kept)
                print(f"[ep {ep}] iter {it}/{len(dl)} loss={avg:.4f} rank={avgr:.4f} kept={total_kept}")

        weights = {
            "model": gmodel.state_dict(),
            "reasoner": reasoner.state_dict(),
            "meta": {
                "geo_root": geo_root,
                "dem_cache_root": dem_cache_root,
                "topk": topk,
                "cell_size_deg": cell_size_deg,
                "freeze_gmodel": freeze_gmodel,
            },
        }
        out_p = os.path.join(run_dir, f"weights_epoch{ep}.pth")
        torch.save(weights, out_p)
        ep_avg = total / max(1, total_kept)
        print(f"saved weights: {out_p}  avg={ep_avg:.4f}")


if __name__ == "__main__":
    main()
