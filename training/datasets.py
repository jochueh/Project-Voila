import os
import re
import math
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from collections import OrderedDict

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import rasterio
from rasterio.windows import Window


_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _default_img_tf(img_size_hw: Tuple[int, int]):
    h, w = int(img_size_hw[0]), int(img_size_hw[1])
    return T.Compose(
        [
            T.Resize((h, w)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def _wrap_lon(lon: float) -> float:
    return ((lon + 180.0) % 360.0) - 180.0


def _grid_encode(lat: np.ndarray, lon: np.ndarray, lat_bins: int, lon_bins: int):
    lat = np.asarray(lat, dtype=np.float32)
    lon = np.asarray(lon, dtype=np.float32)

    lat = np.clip(lat, -89.999999, 89.999999)
    lon = ((lon + 180.0) % 360.0) - 180.0

    lat_step = 180.0 / float(lat_bins)
    lon_step = 360.0 / float(lon_bins)

    i_lat = np.floor((lat + 90.0) / lat_step).astype(np.int64)
    i_lon = np.floor((lon + 180.0) / lon_step).astype(np.int64)

    i_lat = np.clip(i_lat, 0, lat_bins - 1)
    i_lon = np.clip(i_lon, 0, lon_bins - 1)

    cell_id = i_lat * lon_bins + i_lon

    lat_c = -90.0 + (i_lat.astype(np.float32) + 0.5) * lat_step
    lon_c = -180.0 + (i_lon.astype(np.float32) + 0.5) * lon_step

    off_lat = (lat - lat_c) / (lat_step * 0.5)
    off_lon = (lon - lon_c) / (lon_step * 0.5)

    w = np.ones_like(off_lat, dtype=np.float32)
    return i_lat, i_lon, cell_id.astype(np.int64), off_lat.astype(np.float32), off_lon.astype(np.float32), w


def _cell_center(cell_id: int, lat_bins: int, lon_bins: int) -> Tuple[float, float]:
    lat_step = 180.0 / float(lat_bins)
    lon_step = 360.0 / float(lon_bins)
    i_lat = int(cell_id) // int(lon_bins)
    i_lon = int(cell_id) % int(lon_bins)
    lat_c = -90.0 + (float(i_lat) + 0.5) * lat_step
    lon_c = -180.0 + (float(i_lon) + 0.5) * lon_step
    return float(lat_c), float(lon_c)


def _extract_candidate_id_from_stem(stem: str) -> List[str]:
    out = [stem]
    if "_" in stem:
        out.append(stem.split("_")[-1])
    m = re.search(r"[-+]?\d+(?:\.\d+)?_[-+]?\d+(?:\.\d+)?_(.+)$", stem)
    if m:
        out.append(m.group(1))
    return out


def _build_id_to_relpath(images_root: Path, ids_set: set) -> Dict[str, str]:
    id2rel: Dict[str, str] = {}
    for p in images_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in _IMG_EXTS:
            continue
        stem = p.stem
        for cand in _extract_candidate_id_from_stem(stem):
            if cand in ids_set and cand not in id2rel:
                try:
                    id2rel[cand] = str(p.relative_to(images_root)).lstrip("/")
                except Exception:
                    id2rel[cand] = str(p).lstrip("/")
                break
    return id2rel


class EvalGroundDataset(Dataset):
    def __init__(
        self,
        root_processed: str,
        img_size_hw: Tuple[int, int] = (224, 224),
        sample_limit: Optional[int] = None,
        images_subdir: str = "images",
        metadata_csv: str = "metadata.csv",
        seed: int = 42,
    ):
        self.root_processed = str(root_processed)
        self.images_root = Path(self.root_processed) / images_subdir
        self.meta_csv_path = Path(self.root_processed) / metadata_csv
        self.img_tf = _default_img_tf(img_size_hw)

        if not self.meta_csv_path.exists():
            raise FileNotFoundError(f"missing metadata csv: {self.meta_csv_path}")

        df = pd.read_csv(self.meta_csv_path, low_memory=False, dtype={"id": "string"})
        req_cols = {"id", "latitude", "longitude"}
        if not req_cols.issubset(set(df.columns)):
            raise ValueError(f"metadata.csv must have columns {req_cols}, got {set(df.columns)}")

        df["id"] = df["id"].astype("string").fillna("").astype(str).str.strip()
        df = df[df["id"] != ""]
        df = df.dropna(subset=["latitude", "longitude"])

        if sample_limit is not None and int(sample_limit) > 0:
            df = df.sample(n=min(int(sample_limit), len(df)), random_state=int(seed)).reset_index(drop=True)

        self.df = df.reset_index(drop=True)
        ids_set = set(self.df["id"].tolist())
        self.id2rel = _build_id_to_relpath(self.images_root, ids_set)

        keep = []
        for i, r in self.df.iterrows():
            if str(r["id"]) in self.id2rel:
                keep.append(i)
        self.df = self.df.iloc[keep].reset_index(drop=True)

        if len(self.df) == 0:
            raise RuntimeError("no usable samples found (metadata ids did not match any files under images_root)")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        r = self.df.iloc[int(idx)]
        sid = str(r["id"])
        rel = self.id2rel[sid]
        img_path = self.images_root / rel
        img = Image.open(img_path).convert("RGB")
        x = self.img_tf(img)
        lat = float(r["latitude"])
        lon = float(r["longitude"])
        return {"image": x, "lat": lat, "lon": lon, "id": sid}


def _tile_code_for_latlon(lat: float, lon: float) -> str:
    lat = float(lat)
    lon = float(lon)
    if lat >= 0:
        ns = "N"
        lat_deg = int(math.floor(lat))
    else:
        ns = "S"
        lat_deg = int(abs(math.ceil(lat)))
    if lon >= 0:
        ew = "E"
        lon_deg = int(math.floor(lon))
    else:
        ew = "W"
        lon_deg = int(abs(math.ceil(lon)))
    return f"{ns}{lat_deg:02d}{ew}{lon_deg:03d}"


def _index_dem_tiles(dem_root: str) -> Dict[str, str]:
    dem_root_p = Path(dem_root)
    if not dem_root_p.exists():
        return {}
    out: Dict[str, str] = {}
    for p in dem_root_p.rglob("*_dem.tif"):
        name = p.name
        m = re.search(r"([NS]\d{2}[EW]\d{3})_dem\.tif$", name)
        if not m:
            continue
        code = m.group(1)
        out[code] = str(p)
    return out


class RasterioDEMBackend:
    def __init__(self, dem_root: str, max_open: int = 32):
        self.dem_root = str(dem_root)
        self.tile_index = _index_dem_tiles(self.dem_root)
        self.max_open = int(max(1, max_open))
        self._open: "OrderedDict[str, rasterio.io.DatasetReader]" = OrderedDict()

    def close(self):
        for _, ds in list(self._open.items()):
            try:
                ds.close()
            except Exception:
                pass
        self._open.clear()

    def _ds(self, path: str):
        ds = self._open.get(path)
        if ds is not None:
            self._open.move_to_end(path)
            return ds
        ds = rasterio.open(path)
        self._open[path] = ds
        self._open.move_to_end(path)
        while len(self._open) > self.max_open:
            old_path, old_ds = self._open.popitem(last=False)
            try:
                old_ds.close()
            except Exception:
                pass
        return ds

    def has_tile(self, lat: float, lon: float) -> bool:
        code = _tile_code_for_latlon(lat, lon)
        return code in self.tile_index

    def get_window(self, lat: float, lon: float, size_hw: Tuple[int, int]) -> Optional[np.ndarray]:
        code = _tile_code_for_latlon(lat, lon)
        path = self.tile_index.get(code)
        if path is None:
            return None
        ds = self._ds(path)
        lon = _wrap_lon(float(lon))
        lat = float(lat)

        h, w = int(size_hw[0]), int(size_hw[1])
        row, col = ds.index(lon, lat)
        row_off = int(row) - h // 2
        col_off = int(col) - w // 2

        win = Window(col_off=col_off, row_off=row_off, width=w, height=h)
        arr = ds.read(1, window=win, boundless=True, fill_value=np.nan).astype(np.float32)
        if np.isnan(arr).any():
            arr = np.nan_to_num(arr, nan=0.0)
        return arr


class DemCacheBackend:
    def __init__(self, cache_root: str, lat_bins: int, lon_bins: int):
        self.cache_root = str(cache_root)
        self.lat_bins = int(lat_bins)
        self.lon_bins = int(lon_bins)

        cell_ids_path = Path(self.cache_root) / "cell_ids.npy"
        dem_path = Path(self.cache_root) / "dem.npy"

        if not cell_ids_path.exists() or not dem_path.exists():
            raise FileNotFoundError(f"dem cache missing cell_ids.npy or dem.npy under {self.cache_root}")

        self.cell_ids = np.load(str(cell_ids_path), allow_pickle=False).astype(np.int64)
        self.dem = np.load(str(dem_path), mmap_mode="r", allow_pickle=False)

        order = np.argsort(self.cell_ids)
        self.cell_ids_sorted = self.cell_ids[order]
        self.order = order

        uniq, counts = np.unique(self.cell_ids_sorted, return_counts=True)
        self.uniq = uniq
        self.counts = counts
        ends = np.cumsum(counts)
        self.starts = ends - counts

        shape = getattr(self.dem, "shape", None)
        if shape is None or len(shape) != 3:
            raise ValueError(f"dem.npy must be [N,H,W], got {shape}")

        self.patch_h = int(self.dem.shape[1])
        self.patch_w = int(self.dem.shape[2])

    def has_tile(self, lat: float, lon: float) -> bool:
        _, _, cid, _, _, _ = _grid_encode(
            np.asarray([float(lat)], dtype=np.float32),
            np.asarray([float(lon)], dtype=np.float32),
            self.lat_bins,
            self.lon_bins,
        )
        cid = int(cid[0])
        pos = np.searchsorted(self.uniq, cid)
        return bool(pos < len(self.uniq) and int(self.uniq[pos]) == cid)

    def _pick_index_for_cell(self, cell_id: int) -> Optional[int]:
        cell_id = int(cell_id)
        pos = np.searchsorted(self.uniq, cell_id)
        if pos >= len(self.uniq) or int(self.uniq[pos]) != cell_id:
            return None
        s = int(self.starts[pos])
        j = s
        return int(self.order[j])

    def _fit_patch(self, patch: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
        h, w = int(size_hw[0]), int(size_hw[1])
        ph, pw = int(patch.shape[0]), int(patch.shape[1])

        if (ph, pw) == (h, w):
            return patch.astype(np.float32)

        out = np.zeros((h, w), dtype=np.float32)

        ch0 = max(0, (ph - h) // 2)
        cw0 = max(0, (pw - w) // 2)
        patch_c = patch[ch0 : ch0 + min(h, ph), cw0 : cw0 + min(w, pw)]

        oh0 = max(0, (h - patch_c.shape[0]) // 2)
        ow0 = max(0, (w - patch_c.shape[1]) // 2)
        out[oh0 : oh0 + patch_c.shape[0], ow0 : ow0 + patch_c.shape[1]] = patch_c.astype(np.float32)
        return out

    def get_window(self, lat: float, lon: float, size_hw: Tuple[int, int]) -> Optional[np.ndarray]:
        lat = float(lat)
        lon = float(lon)
        _, _, cid, _, _, _ = _grid_encode(
            np.asarray([lat], dtype=np.float32),
            np.asarray([lon], dtype=np.float32),
            self.lat_bins,
            self.lon_bins,
        )
        cid = int(cid[0])
        idx = self._pick_index_for_cell(cid)
        if idx is None:
            return None
        patch = np.asarray(self.dem[idx], dtype=np.float32)
        return self._fit_patch(patch, size_hw)


def build_dem_backend(dem_root: str, dem_cfg: dict, lat_bins: int = 36, lon_bins: int = 72):
    cache_root = str(dem_cfg.get("cache_root", "") or dem_cfg.get("dem_cache_root", "") or "")
    if cache_root:
        try:
            return DemCacheBackend(cache_root=cache_root, lat_bins=lat_bins, lon_bins=lon_bins)
        except Exception:
            max_open = int(dem_cfg.get("max_open_files", 32))
            return RasterioDEMBackend(dem_root=dem_root, max_open=max_open)
    max_open = int(dem_cfg.get("max_open_files", 32))
    return RasterioDEMBackend(dem_root=dem_root, max_open=max_open)


class GroundDEMDataset(Dataset):
    def __init__(
        self,
        root_processed: str,
        dem_root: Optional[str] = None,
        lat_bins: int = 36,
        lon_bins: int = 72,
        img_size_hw: Tuple[int, int] = (224, 224),
        dem_size_hw: Tuple[int, int] = (64, 64),
        k_dem: int = 1,
        sample_limit: Optional[int] = None,
        shuffle: bool = True,
        images_subdir: str = "images",
        metadata_csv: str = "metadata.csv",
        seed: int = 42,
        dem_backend=None,
        missing_dem_policy: str = "skip",
        **kwargs,
    ):
        self.root_processed = str(root_processed)
        self.images_root = Path(self.root_processed) / images_subdir
        self.meta_csv_path = Path(self.root_processed) / metadata_csv
        self.img_tf = _default_img_tf(img_size_hw)
        self.lat_bins = int(lat_bins)
        self.lon_bins = int(lon_bins)
        self.dem_backend = dem_backend
        self.missing_dem_policy = str(missing_dem_policy)
        self.dem_size_hw = (int(dem_size_hw[0]), int(dem_size_hw[1]))
        self.k_dem = int(k_dem)
        self.dem_root = str(dem_root) if dem_root is not None else ""

        if not self.meta_csv_path.exists():
            raise FileNotFoundError(f"missing metadata csv: {self.meta_csv_path}")

        df = pd.read_csv(self.meta_csv_path, low_memory=False, dtype={"id": "string"})
        req_cols = {"id", "latitude", "longitude"}
        if not req_cols.issubset(set(df.columns)):
            raise ValueError(f"metadata.csv must have columns {req_cols}, got {set(df.columns)}")

        df["id"] = df["id"].astype("string").fillna("").astype(str).str.strip()
        df = df[df["id"] != ""]
        df = df.dropna(subset=["latitude", "longitude"])

        if sample_limit is not None and int(sample_limit) > 0:
            df = df.sample(n=min(int(sample_limit), len(df)), random_state=int(seed)).reset_index(drop=True)

        if shuffle:
            df = df.sample(frac=1.0, random_state=int(seed)).reset_index(drop=True)

        self.df = df.reset_index(drop=True)
        ids_set = set(self.df["id"].tolist())
        self.id2rel = _build_id_to_relpath(self.images_root, ids_set)

        keep = []
        for i, r in self.df.iterrows():
            sid = str(r["id"])
            if sid not in self.id2rel:
                continue
            if self.dem_backend is not None and self.missing_dem_policy == "skip":
                lat = float(r["latitude"])
                lon = float(r["longitude"])
                if hasattr(self.dem_backend, "has_tile"):
                    if not bool(self.dem_backend.has_tile(lat, lon)):
                        continue
            keep.append(i)

        self.df = self.df.iloc[keep].reset_index(drop=True)

        if len(self.df) == 0:
            raise RuntimeError("no usable samples found after filtering")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        r = self.df.iloc[int(idx)]
        sid = str(r["id"])
        rel = self.id2rel[sid]
        img_path = self.images_root / rel
        img = Image.open(img_path).convert("RGB")
        x = self.img_tf(img)
        lat = float(r["latitude"])
        lon = float(r["longitude"])

        out = {"image": x, "lat": lat, "lon": lon, "id": sid}

        if self.dem_backend is not None:
            dem = self.dem_backend.get_window(lat=lat, lon=lon, size_hw=self.dem_size_hw)
            if dem is None:
                if self.missing_dem_policy == "zero":
                    dem = np.zeros(self.dem_size_hw, dtype=np.float32)
                elif self.missing_dem_policy == "nan":
                    dem = np.full(self.dem_size_hw, np.nan, dtype=np.float32)
                else:
                    dem = np.zeros(self.dem_size_hw, dtype=np.float32)
            out["dem"] = torch.from_numpy(np.asarray(dem, dtype=np.float32))
        return out
