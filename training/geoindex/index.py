import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def _first_existing(root: Path, names: List[str]) -> Optional[Path]:
    for n in names:
        p = root / n
        if p.exists():
            return p
    return None


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n


def _load_memmap_f16(path: Path, n: int, d: int) -> np.memmap:
    mm = np.memmap(str(path), dtype=np.float16, mode="r")
    if mm.size != n * d:
        raise ValueError(f"bad f16 memmap size: {path} got={mm.size} expected={n*d}")
    return mm.reshape((n, d))


class GeoIndex:
    def __init__(self, root: str):
        self.root = str(root)
        r = Path(self.root)
        if not r.exists():
            raise FileNotFoundError(f"GeoIndex root does not exist: {self.root}")

        self.faiss_available = False
        self.index = None
        self.idx_path = str(r / "faiss.index")

        self.mode = ""
        self.D = 0

        cell_ids_path = _first_existing(r, ["cell_ids.npy", "cells.npy", "cell_id.npy"])
        cell_emb_path = _first_existing(
            r,
            [
                "cell_embeds.npy",
                "cell_embeddings.npy",
                "cell_vectors.npy",
                "centroids.npy",
                "cell_embeds.f16",
                "cell_embeddings.f16",
                "cell_vectors.f16",
                "centroids.f16",
            ],
        )

        if cell_ids_path is not None and cell_emb_path is not None:
            self.cell_ids = np.load(str(cell_ids_path), allow_pickle=False).astype(np.int64)

            if cell_emb_path.suffix.lower() == ".npy":
                emb = np.load(str(cell_emb_path), mmap_mode=None, allow_pickle=False)
                emb = np.asarray(emb)
            else:
                mm = np.memmap(str(cell_emb_path), dtype=np.float16, mode="r")
                if mm.size % int(self.cell_ids.shape[0]) != 0:
                    raise ValueError(f"cell embedding f16 size mismatch: {cell_emb_path}")
                d = int(mm.size // int(self.cell_ids.shape[0]))
                emb = np.asarray(mm.reshape((int(self.cell_ids.shape[0]), d)))

            if emb.ndim != 2 or int(emb.shape[0]) != int(self.cell_ids.shape[0]):
                raise ValueError(f"cell embeddings shape mismatch: {cell_emb_path} shape={emb.shape}")

            self.embeds = _l2_normalize(np.asarray(emb, dtype=np.float32))
            self.D = int(self.embeds.shape[1])
            self.mode = "cells"
            self._init_faiss_cell()
            return

        emb_npy = _first_existing(r, ["embeddings.npy"])
        emb_f16 = _first_existing(r, ["embeddings.f16", "embeddings.fp16", "embeddings.memmap.f16"])
        meta_npz = _first_existing(r, ["sample_meta.npz"])
        ids_npy = _first_existing(r, ["ids.npy", "cell_id.npy", "cell_ids.npy"])

        if emb_npy is None and emb_f16 is None:
            raise FileNotFoundError(
                f"GeoIndex missing embeddings (expected embeddings.npy or embeddings.f16) in {self.root}"
            )

        if meta_npz is None and ids_npy is None:
            raise FileNotFoundError(
                f"GeoIndex missing sample->cell ids (expected sample_meta.npz or ids.npy) in {self.root}"
            )

        if meta_npz is not None:
            z = np.load(str(meta_npz), allow_pickle=False)
            if "cell_id" not in z:
                raise ValueError(f"sample_meta.npz missing 'cell_id' in {meta_npz}")
            self.sample_cell_id = np.asarray(z["cell_id"]).astype(np.int64)
        else:
            self.sample_cell_id = np.load(str(ids_npy), allow_pickle=False).astype(np.int64)

        if emb_npy is not None:
            X = np.load(str(emb_npy), mmap_mode="r", allow_pickle=False)
            X = np.asarray(X)
            if X.ndim != 2:
                raise ValueError(f"embeddings.npy must be 2D, got {X.shape}")
            self.N = int(X.shape[0])
            self.D = int(X.shape[1])
            if int(self.sample_cell_id.shape[0]) != self.N:
                raise ValueError(
                    f"sample_cell_id length {self.sample_cell_id.shape[0]} != embeddings rows {self.N}"
                )
            self.sample_emb = X
            self.sample_emb_is_f16 = False
        else:
            d_guess = None
            if Path(self.idx_path).exists():
                d_guess = None
            if d_guess is None:
                d_guess = 256
            n = int(self.sample_cell_id.shape[0])
            X = _load_memmap_f16(emb_f16, n=n, d=int(d_guess))
            self.N = n
            self.D = int(X.shape[1])
            self.sample_emb = X
            self.sample_emb_is_f16 = True

        self.mode = "samples"
        self._init_faiss_sample()

    def _init_faiss_cell(self) -> None:
        try:
            import faiss 

            self.faiss_available = True
            if os.path.exists(self.idx_path):
                self.index = faiss.read_index(self.idx_path)
                return
            idx = faiss.IndexFlatIP(self.D)
            idx.add(self.embeds.astype(np.float32, copy=False))
            faiss.write_index(idx, self.idx_path)
            self.index = idx
        except Exception:
            self.faiss_available = False
            self.index = None

    def _init_faiss_sample(self) -> None:
        try:
            import faiss 

            self.faiss_available = True
            if os.path.exists(self.idx_path):
                self.index = faiss.read_index(self.idx_path)
                return

            idx = faiss.IndexFlatIP(self.D)

            if self.sample_emb_is_f16:
                bs = 200000
                for s in range(0, self.N, bs):
                    e = min(self.N, s + bs)
                    chunk = np.asarray(self.sample_emb[s:e], dtype=np.float32)
                    chunk = _l2_normalize(chunk)
                    idx.add(chunk.astype(np.float32, copy=False))
            else:
                chunk = _l2_normalize(np.asarray(self.sample_emb, dtype=np.float32))
                idx.add(chunk.astype(np.float32, copy=False))

            faiss.write_index(idx, self.idx_path)
            self.index = idx
        except Exception:
            self.faiss_available = False
            self.index = None

    def _search_samples_numpy(self, qn: np.ndarray, topk_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        X = self.sample_emb
        if self.sample_emb_is_f16:
            X32 = np.asarray(X, dtype=np.float32)
            X32 = _l2_normalize(X32)
            sims = qn @ X32.T
        else:
            X32 = np.asarray(X, dtype=np.float32)
            X32 = _l2_normalize(X32)
            sims = qn @ X32.T

        k = min(int(topk_samples), int(X32.shape[0]))
        idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        row = np.arange(idx.shape[0])[:, None]
        sel = sims[row, idx]
        order = np.argsort(-sel, axis=1)
        idx = idx[row, order]
        scores = sims[row, idx]
        return scores.astype(np.float32), idx.astype(np.int64)

    def search(self, q: np.ndarray, topk: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        q = np.asarray(q, dtype=np.float32)
        if q.ndim == 1:
            q = q[None, :]
        if q.ndim != 2 or int(q.shape[1]) != int(self.D):
            raise ValueError(f"query must be [N,{self.D}], got {q.shape}")
        qn = _l2_normalize(q)

        k = int(topk)
        if k <= 0:
            raise ValueError("topk must be > 0")

        if self.mode == "cells":
            k2 = min(k, int(self.embeds.shape[0]))
            if self.index is not None and self.faiss_available:
                scores, idx = self.index.search(qn.astype(np.float32, copy=False), k2)
                cell_ids = self.cell_ids[idx]
                return scores.astype(np.float32), cell_ids.astype(np.int64)

            sims = qn @ self.embeds.T
            idx = np.argpartition(-sims, kth=k2 - 1, axis=1)[:, :k2]
            row = np.arange(idx.shape[0])[:, None]
            sel = sims[row, idx]
            order = np.argsort(-sel, axis=1)
            idx = idx[row, order]
            scores = sims[row, idx]
            cell_ids = self.cell_ids[idx]
            return scores.astype(np.float32), cell_ids.astype(np.int64)

        topk_samples = min(self.N, max(k * 20, 200))
        if self.index is not None and self.faiss_available:
            s_scores, s_idx = self.index.search(qn.astype(np.float32, copy=False), topk_samples)
            s_scores = s_scores.astype(np.float32)
            s_idx = s_idx.astype(np.int64)
        else:
            s_scores, s_idx = self._search_samples_numpy(qn, topk_samples)

        out_scores = np.full((qn.shape[0], k), -1e9, dtype=np.float32)
        out_cells = np.full((qn.shape[0], k), -1, dtype=np.int64)

        for i in range(qn.shape[0]):
            seen = {}
            filled = 0
            for j in range(s_idx.shape[1]):
                si = int(s_idx[i, j])
                cid = int(self.sample_cell_id[si])
                sc = float(s_scores[i, j])
                if cid in seen:
                    if sc > seen[cid]:
                        seen[cid] = sc
                    continue
                seen[cid] = sc
                out_cells[i, filled] = cid
                out_scores[i, filled] = sc
                filled += 1
                if filled >= k:
                    break

            if filled < k and filled > 0:
                out_cells[i, filled:] = out_cells[i, filled - 1]
                out_scores[i, filled:] = out_scores[i, filled - 1]

        return out_scores, out_cells
