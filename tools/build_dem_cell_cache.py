import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml


def _read_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _wrap_lon(lon: float) -> float:
    x = ((lon + 180.0) % 360.0) - 180.0
    if x == 180.0:
        x = -180.0
    return x


def tile_name_for_latlon(lat: float, lon: float) -> str:
    lat = float(np.clip(lat, -89.999999, 89.999999))
    lon = float(_wrap_lon(lon))
    lat0 = int(np.floor(lat))
    lon0 = int(np.floor(lon))
    ns = "N" if lat0 >= 0 else "S"
    ew = "E" if lon0 >= 0 else "W"
    return f"ASTGTMV003_{ns}{abs(lat0):02d}{ew}{abs(lon0):03d}"


def tile_bounds(tile: str) -> tuple[float, float, float, float]:
    base = tile
    if base.startswith("ASTGTMV003_"):
        base = base[len("ASTGTMV003_") :]
    ns = base[0]
    lat_deg = int(base[1:3])
    ew = base[3]
    lon_deg = int(base[4:7])

    south = float(lat_deg if ns == "N" else -lat_deg)
    west = float(lon_deg if ew == "E" else -lon_deg)
    north = south + 1.0
    east = west + 1.0
    return west, east, south, north


def _read_tif(path: Path) -> np.ndarray:
    try:
        import tifffile

        arr = tifffile.imread(str(path))
    except Exception:
        try:
            import rasterio

            with rasterio.open(str(path)) as ds:
                arr = ds.read(1)
        except Exception:
            from PIL import Image

            arr = np.asarray(Image.open(path))
    if arr.ndim > 2:
        arr = arr[..., 0]
    return arr


def _latlon_to_rc(lat: float, lon: float, west: float, east: float, south: float, north: float, h: int, w: int) -> tuple[int, int]:
    lat = max(min(lat, north - 1e-9), south + 1e-9)
    lon = max(min(lon, east - 1e-9), west + 1e-9)
    y = (north - lat) / (north - south)
    x = (lon - west) / (east - west)
    r = int(round(y * (h - 1)))
    c = int(round(x * (w - 1)))
    r = max(0, min(h - 1, r))
    c = max(0, min(w - 1, c))
    return r, c


def _extract_patch(dem: np.ndarray, valid: np.ndarray, r: int, c: int, ph: int, pw: int) -> np.ndarray:
    rh = ph // 2
    rw = pw // 2
    r0 = r - rh
    r1 = r0 + ph
    c0 = c - rw
    c1 = c0 + pw

    pad_top = max(0, -r0)
    pad_left = max(0, -c0)
    pad_bot = max(0, r1 - dem.shape[0])
    pad_right = max(0, c1 - dem.shape[1])

    rr0 = max(0, r0)
    rr1 = min(dem.shape[0], r1)
    cc0 = max(0, c0)
    cc1 = min(dem.shape[1], c1)

    patch = dem[rr0:rr1, cc0:cc1]
    v = valid[rr0:rr1, cc0:cc1]

    if pad_top or pad_bot or pad_left or pad_right:
        patch = np.pad(patch, ((pad_top, pad_bot), (pad_left, pad_right)), mode="constant", constant_values=0)
        v = np.pad(v, ((pad_top, pad_bot), (pad_left, pad_right)), mode="constant", constant_values=False)

    if patch.shape != (ph, pw):
        tmp = np.zeros((ph, pw), dtype=patch.dtype)
        hh = min(ph, patch.shape[0])
        ww = min(pw, patch.shape[1])
        tmp[:hh, :ww] = patch[:hh, :ww]
        patch = tmp
        tmpv = np.zeros((ph, pw), dtype=bool)
        tmpv[:hh, :ww] = v[:hh, :ww]
        v = tmpv

    out = np.zeros((ph, pw), dtype=np.float32)
    if np.any(v):
        vals = patch[v].astype(np.float32)
        m = float(vals.mean())
        s = float(vals.std())
        if not np.isfinite(s) or s < 1e-6:
            s = 1.0
        z = (patch.astype(np.float32) - m) / s
        z = np.clip(z, -3.0, 3.0) / 3.0
        out[v] = z[v]
    return out


def _maybe_remove(path: Path):
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def _tile_base_from_path(p: Path) -> str | None:
    name = p.name
    lo = name.lower()
    if not (lo.endswith(".tif") or lo.endswith(".tiff")):
        return None
    stem = p.stem
    if stem.lower().endswith("_dem"):
        stem = stem[: -4]
    elif stem.lower().endswith("_num"):
        return None
    if stem.startswith("ASTGTMV003_"):
        return stem
    if len(stem) == 7 and (stem[0] in "NS") and (stem[3] in "EW"):
        return "ASTGTMV003_" + stem
    return None


def _build_dem_index(dem_root: Path) -> dict[str, str]:
    idx: dict[str, str] = {}
    for p in dem_root.rglob("*"):
        if not p.is_file():
            continue
        base = _tile_base_from_path(p)
        if not base:
            continue
        idx[base] = str(p)
        short = base.replace("ASTGTMV003_", "")
        idx[short] = str(p)
    return idx


def _haversine_km_vec(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    r = 6371.0088
    a1 = np.deg2rad(lat1)
    o1 = np.deg2rad(lon1)
    a2 = np.deg2rad(lat2.astype(np.float64, copy=False))
    o2 = np.deg2rad(lon2.astype(np.float64, copy=False))
    dlat = a2 - a1
    dlon = o2 - o1
    x = np.sin(dlat / 2.0) ** 2 + np.cos(a1) * np.cos(a2) * (np.sin(dlon / 2.0) ** 2)
    c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(x)))
    return r * c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="")
    p.add_argument("--data_cfg", required=True)
    p.add_argument("--geo_root", default="")
    p.add_argument("--dem_root", default="")
    p.add_argument("--out_root", default="")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if args.repo:
        sys.path.append(args.repo)

    data_cfg = _read_yaml(args.data_cfg)

    geo_root = args.geo_root or str((data_cfg.get("geoindex") or {}).get("root", ""))
    if not geo_root:
        raise SystemExit("geo_root missing (set data.yaml geoindex.root or pass --geo_root)")

    dem_root = args.dem_root or str(((data_cfg.get("dem") or {}).get("root", "")))
    if not dem_root:
        raise SystemExit("dem_root missing (set data.yaml dem.root or pass --dem_root)")

    dem_cfg = data_cfg.get("dem") or {}
    dem_size_hw = dem_cfg.get("dem_size_hw", [64, 64])
    ph = int(dem_size_hw[0])
    pw = int(dem_size_hw[1])

    out_root = Path(args.out_root or str(Path(geo_root) / "dem_cache"))
    out_root.mkdir(parents=True, exist_ok=True)

    ids_path = out_root / "cell_ids.npy"
    dem_path = out_root / "dem_patches.npy"
    src_path = out_root / "source_cell_ids.npy"

    if (ids_path.exists() or dem_path.exists() or src_path.exists()) and not args.overwrite:
        raise SystemExit(f"Output exists. Use --overwrite or delete {ids_path}, {dem_path}, {src_path}")

    if args.overwrite:
        _maybe_remove(ids_path)
        _maybe_remove(dem_path)
        _maybe_remove(src_path)
        _maybe_remove(Path(str(dem_path) + ".tmp"))
        _maybe_remove(Path(str(src_path) + ".tmp"))

    cm = np.load(str(Path(geo_root) / "cell_meta.npz"))
    cell_ids = cm["cell_ids"].astype(np.int64, copy=False)
    cell_lat = cm["cell_lat"].astype(np.float32, copy=False)
    cell_lon = cm["cell_lon"].astype(np.float32, copy=False)

    n = int(cell_ids.shape[0])
    if args.limit and args.limit > 0:
        n = min(n, int(args.limit))
        cell_ids = cell_ids[:n]
        cell_lat = cell_lat[:n]
        cell_lon = cell_lon[:n]

    np.save(str(ids_path), cell_ids)

    tmp_dem = Path(str(dem_path) + ".tmp")
    mm_dem = np.memmap(str(tmp_dem), mode="w+", dtype=np.float32, shape=(n, ph, pw))

    tmp_src = Path(str(src_path) + ".tmp")
    mm_src = np.memmap(str(tmp_src), mode="w+", dtype=np.int64, shape=(n,))

    dem_root_p = Path(dem_root)
    if not dem_root_p.exists():
        raise SystemExit(f"dem_root not found: {str(dem_root_p)}")

    dem_index = _build_dem_index(dem_root_p)

    cell_tile = np.array([tile_name_for_latlon(float(cell_lat[i]), float(cell_lon[i])) for i in range(n)], dtype=object)
    tile_present = np.zeros(n, dtype=bool)
    tile_path = np.empty(n, dtype=object)

    for i in range(n):
        t = str(cell_tile[i])
        pth_str = dem_index.get(t) or dem_index.get(t.replace("ASTGTMV003_", ""))
        if pth_str:
            tile_present[i] = True
            tile_path[i] = pth_str
        else:
            tile_present[i] = False
            tile_path[i] = ""

    if not np.any(tile_present):
        raise SystemExit("No DEM tiles found under dem_root for any geoindex cell centroids")

    cand_idx = np.nonzero(tile_present)[0]
    cand_lat = cell_lat[cand_idx].astype(np.float64, copy=False)
    cand_lon = cell_lon[cand_idx].astype(np.float64, copy=False)

    tile_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    t0 = time.time()
    missing_exact = 0
    substituted = 0

    for i in range(n):
        lat = float(cell_lat[i])
        lon = float(cell_lon[i])

        if not np.isfinite(lat) or not np.isfinite(lon):
            mm_dem[i] = 0.0
            mm_src[i] = int(cell_ids[i])
            missing_exact += 1
            continue

        src_i = i
        tile = str(cell_tile[i])
        pth_str = str(tile_path[i]) if tile_present[i] else ""

        if not pth_str:
            missing_exact += 1
            d = _haversine_km_vec(lat, lon, cand_lat, cand_lon)
            j = int(np.argmin(d))
            src_i = int(cand_idx[j])
            tile = str(cell_tile[src_i])
            pth_str = str(tile_path[src_i])
            substituted += 1
            lat = float(cell_lat[src_i])
            lon = float(cell_lon[src_i])

        if tile not in tile_cache:
            arr = _read_tif(Path(pth_str))
            dem = arr.astype(np.float32, copy=False)
            valid = np.isfinite(dem)
            valid &= dem > -9000.0
            dem = np.where(valid, dem, 0.0)
            tile_cache[tile] = (dem, valid)

        dem, valid = tile_cache[tile]
        west, east, south, north = tile_bounds(tile)
        lonw = float(_wrap_lon(lon))
        r, c = _latlon_to_rc(lat, lonw, west, east, south, north, dem.shape[0], dem.shape[1])
        mm_dem[i] = _extract_patch(dem, valid, r, c, ph, pw)
        mm_src[i] = int(cell_ids[src_i])

        if args.log_every and ((i + 1) % args.log_every == 0 or (i + 1) == n):
            dt = max(time.time() - t0, 1e-6)
            rate = (i + 1) / dt
            print(
                f"[dem_cache] {i+1}/{n} | {rate:.1f} cells/s | tiles_cached={len(tile_cache)} | indexed={len(dem_index)} | missing_exact={missing_exact} | substituted={substituted}"
            )

    mm_dem.flush()
    mm_src.flush()
    os.replace(str(tmp_dem), str(dem_path))
    os.replace(str(tmp_src), str(src_path))

    dt = max(time.time() - t0, 1e-6)
    print(
        f"[dem_cache] done: {str(out_root)} | cells={n} | patch={ph}x{pw} | indexed={len(dem_index)} | missing_exact={missing_exact} | substituted={substituted} | {n/dt:.1f} cells/s"
    )


if __name__ == "__main__":
    main()
