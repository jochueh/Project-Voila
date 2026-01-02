import argparse
import os
import subprocess
import sys
import yaml


DEFAULTS = {
    "data_cfg": "configs/data.yaml",
    "grid_cfg": "configs/grid.yaml",
    "model_cfg": "configs/model.yaml",
    "train_gcnn_cfg": "configs/train_ground_dem.yaml",
    "train_phase2_cfg": "configs/train_phase2.yaml",
    "runs_dir": "/mnt/data/checkpoints",
    "eval_sample_limit": 1000,
    "geoindex_limit": 0,
    "phase2_topk": 50,
    "geoindex_root": "/mnt/data/geoindex",
}


def run(cmd, cwd=None):
    print(" ".join(cmd))
    return subprocess.run(cmd, check=True, cwd=cwd)


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_train_gcnn(data_cfg, grid_cfg, model_cfg, train_cfg, runs_dir):
    cmd = [
        sys.executable,
        "-m",
        "training.trainer_gcnn",
        "--data-cfg",
        data_cfg,
        "--grid-cfg",
        grid_cfg,
        "--model-cfg",
        model_cfg,
        "--train-cfg",
        train_cfg,
        "--runs-dir",
        runs_dir,
    ]
    run(cmd)


def run_train_phase2(data_cfg, grid_cfg, model_cfg, train_cfg, runs_dir, ckpt, topk):
    cmd = [
        sys.executable,
        "-m",
        "training.trainer_coarse2fine",
        "--data-cfg",
        data_cfg,
        "--grid-cfg",
        grid_cfg,
        "--model-cfg",
        model_cfg,
        "--train-cfg",
        train_cfg,
        "--runs-dir",
        runs_dir,
        "--topk",
        str(topk),
    ]
    if ckpt:
        cmd += ["--ckpt", ckpt]
    run(cmd)


def run_eval(ckpt_dir, data_cfg, grid_cfg, model_cfg, sample_limit):
    cmd = [
        sys.executable,
        "-m",
        "training.evaluate",
        "--data-cfg",
        data_cfg,
        "--grid-cfg",
        grid_cfg,
        "--model-cfg",
        model_cfg,
        "--ckpt-dir",
        ckpt_dir,
        "--sample-limit",
        str(sample_limit),
    ]
    run(cmd)


def run_eval2(ckpt_dir, data_cfg, grid_cfg, model_cfg, geo_root, dem_cache_root, topk, sample_limit):
    cmd = [
        sys.executable,
        "-m",
        "training.evaluate_phase2",
        "--data-cfg",
        data_cfg,
        "--grid-cfg",
        grid_cfg,
        "--model-cfg",
        model_cfg,
        "--ckpt-dir",
        ckpt_dir,
        "--topk",
        str(topk),
        "--sample-limit",
        str(sample_limit),
    ]
    if geo_root:
        cmd += ["--geo-root", geo_root]
    if dem_cache_root:
        cmd += ["--dem-cache", dem_cache_root]

    print(f"[main] eval2 resolved geo_root={geo_root} dem_cache={dem_cache_root} ckpt_dir={ckpt_dir} topk={topk} sample_limit={sample_limit}")
    run(cmd)


def run_geoindex(repo, images_root, meta_csv, grid_cfg, model_cfg, out_root, ckpt, limit):
    cmd = [
        sys.executable,
        "-m",
        "tools.geoindex",
        "--repo",
        repo,
        "--images_root",
        images_root,
        "--meta",
        meta_csv,
        "--grid_cfg",
        grid_cfg,
        "--model_cfg",
        model_cfg,
        "--out_root",
        out_root,
        "--ckpt",
        ckpt,
    ]
    if limit and int(limit) > 0:
        cmd += ["--limit", str(int(limit))]
    run(cmd)


def run_dem_cache(repo, data_cfg, geo_root, dem_root, out_root, limit, log_every, overwrite):
    cmd = [
        sys.executable,
        "-m",
        "tools.build_dem_cell_cache",
        "--repo",
        repo,
        "--data_cfg",
        data_cfg,
    ]
    if geo_root:
        cmd += ["--geo_root", geo_root]
    if dem_root:
        cmd += ["--dem_root", dem_root]
    if out_root:
        cmd += ["--out_root", out_root]
    if limit and int(limit) > 0:
        cmd += ["--limit", str(int(limit))]
    if log_every and int(log_every) > 0:
        cmd += ["--log_every", str(int(log_every))]
    if overwrite:
        cmd += ["--overwrite"]
    run(cmd)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("train", help="phase 1 ground-only gcnn training")
    p1.add_argument("--data-cfg", default=DEFAULTS["data_cfg"])
    p1.add_argument("--grid-cfg", default=DEFAULTS["grid_cfg"])
    p1.add_argument("--model-cfg", default=DEFAULTS["model_cfg"])
    p1.add_argument("--train-cfg", default=DEFAULTS["train_gcnn_cfg"])
    p1.add_argument("--runs-dir", default=DEFAULTS["runs_dir"])

    p2 = sub.add_parser("phase2", help="phase 2 coarse2fine DEM reasoner training")
    p2.add_argument("--data-cfg", default=DEFAULTS["data_cfg"])
    p2.add_argument("--grid-cfg", default=DEFAULTS["grid_cfg"])
    p2.add_argument("--model-cfg", default=DEFAULTS["model_cfg"])
    p2.add_argument("--train-cfg", default=DEFAULTS["train_phase2_cfg"])
    p2.add_argument("--runs-dir", default=DEFAULTS["runs_dir"])
    p2.add_argument("--ckpt", default="")
    p2.add_argument("--topk", type=int, default=DEFAULTS["phase2_topk"])

    pe = sub.add_parser("eval", help="evaluate checkpoint directory (phase 1)")
    pe.add_argument("--ckpt-dir", required=True)
    pe.add_argument("--data-cfg", default=DEFAULTS["data_cfg"])
    pe.add_argument("--grid-cfg", default=DEFAULTS["grid_cfg"])
    pe.add_argument("--model-cfg", default=DEFAULTS["model_cfg"])
    pe.add_argument("--sample-limit", type=int, default=DEFAULTS["eval_sample_limit"])

    pe2 = sub.add_parser("eval2", help="evaluate checkpoint directory with phase 2 retrieval+DEM reasoning")
    pe2.add_argument("--ckpt-dir", required=True)
    pe2.add_argument("--data-cfg", default=DEFAULTS["data_cfg"])
    pe2.add_argument("--grid-cfg", default=DEFAULTS["grid_cfg"])
    pe2.add_argument("--model-cfg", default=DEFAULTS["model_cfg"])
    pe2.add_argument("--geo-root", default="")
    pe2.add_argument("--dem-cache", default="")
    pe2.add_argument("--topk", type=int, default=DEFAULTS["phase2_topk"])
    pe2.add_argument("--sample-limit", type=int, default=DEFAULTS["eval_sample_limit"])

    pg = sub.add_parser("geoindex", help="build geoindex from a gcnn checkpoint")
    pg.add_argument("--repo", default=os.getcwd())
    pg.add_argument("--images-root", required=True)
    pg.add_argument("--meta", required=True)
    pg.add_argument("--grid-cfg", default=DEFAULTS["grid_cfg"])
    pg.add_argument("--model-cfg", default=DEFAULTS["model_cfg"])
    pg.add_argument("--out-root", default=DEFAULTS["geoindex_root"])
    pg.add_argument("--ckpt", required=True)
    pg.add_argument("--limit", type=int, default=DEFAULTS["geoindex_limit"])

    pd = sub.add_parser("dem-cache", help="build DEM cell cache for phase 2")
    pd.add_argument("--repo", default=os.getcwd())
    pd.add_argument("--data-cfg", default=DEFAULTS["data_cfg"])
    pd.add_argument("--geo-root", default="") # will be auto-filled below if empty
    pd.add_argument("--dem-root", default="")
    pd.add_argument("--out-root", default="")
    pd.add_argument("--limit", type=int, default=0)
    pd.add_argument("--log-every", type=int, default=50)
    pd.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    if args.cmd == "train":
        run_train_gcnn(args.data_cfg, args.grid_cfg, args.model_cfg, args.train_cfg, args.runs_dir)
        return

    if args.cmd == "phase2":
        run_train_phase2(args.data_cfg, args.grid_cfg, args.model_cfg, args.train_cfg, args.runs_dir, args.ckpt, args.topk)
        return

    if args.cmd == "eval":
        run_eval(args.ckpt_dir, args.data_cfg, args.grid_cfg, args.model_cfg, args.sample_limit)
        return

    if args.cmd == "eval2":
        data_cfg = load_yaml(args.data_cfg)
        if not args.geo_root:
            args.geo_root = str((data_cfg.get("geoindex", {}) or {}).get("root", DEFAULTS["geoindex_root"]))
        if not args.dem_cache:
            args.dem_cache = str((data_cfg.get("dem_cache", {}) or {}).get("root", os.path.join(args.geo_root, "dem_cache")))
        run_eval2(args.ckpt_dir, args.data_cfg, args.grid_cfg, args.model_cfg, args.geo_root, args.dem_cache, args.topk, args.sample_limit)
        return

    if args.cmd == "geoindex":
        run_geoindex(args.repo, args.images_root, args.meta, args.grid_cfg, args.model_cfg, args.out_root, args.ckpt, args.limit)
        return

    if args.cmd == "dem-cache":
        data_cfg = load_yaml(args.data_cfg)
        if not args.geo_root:
            args.geo_root = str((data_cfg.get("geoindex", {}) or {}).get("root", DEFAULTS["geoindex_root"]))
        run_dem_cache(args.repo, args.data_cfg, args.geo_root, args.dem_root, args.out_root, args.limit, args.log_every, args.overwrite)
        return


if __name__ == "__main__":
    main()
