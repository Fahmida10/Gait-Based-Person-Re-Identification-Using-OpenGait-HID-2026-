#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List


def _require_yaml():
    import yaml  # type: ignore
    return yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    yaml = _require_yaml()
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, data: Dict[str, Any]) -> None:
    yaml = _require_yaml()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _resolve_partition_path(opengait_root: Path, partition_value: str) -> Path:
    p = Path(partition_value).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (opengait_root / p).resolve()


def _enabled_stages_in_order(stages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    enabled = [s for s in stages if s.get("enabled", True)]
    return sorted(enabled, key=lambda s: (int(s.get("stage", 10_000)), str(s.get("name", ""))))


def validate_user_paths(user_cfg: Dict[str, Any]) -> None:
    project = user_cfg["project"]
    opengait_root = Path(project["opengait_root"]).expanduser().resolve()
    ckpt = Path(user_cfg["checkpoint"]["pretrained_ckpt"]).expanduser().resolve()
    errs: List[str] = []
    if not opengait_root.is_dir():
        errs.append(f"OpenGait root missing: {opengait_root}")
    if not ckpt.is_file():
        errs.append(f"Pretrained checkpoint missing: {ckpt}")
    for st in _enabled_stages_in_order(user_cfg["datasets"]):
        ds_root = Path(st["dataset_root"]).expanduser().resolve()
        ds_part = _resolve_partition_path(opengait_root, st["dataset_partition"])
        if not ds_root.is_dir():
            errs.append(f"[{st['name']}] dataset_root missing: {ds_root}")
        if not ds_part.is_file():
            errs.append(f"[{st['name']}] dataset_partition missing: {ds_part}")
    if errs:
        raise FileNotFoundError("Path validation failed:\n" + "\n".join(errs))


def _train_transform_cfg(tr: Dict[str, Any]) -> List[Dict[str, Any]]:
    aug = tr.get("augmentation", {})
    use_aug = bool(aug.get("enabled", False))
    base_resize = {
        "type": "ResizeSilTransform",
        "img_h": int(tr["image_size"]),
        "img_w": int(tr["image_size"]),
        "interpolation": "nearest",
    }
    if not use_aug:
        return [base_resize, {"type": "BaseSilCuttingTransform", "img_w": int(tr["image_size"])}]
    return [{
        "type": "Compose",
        "trf_cfg": [
            base_resize,
            {"type": "BaseSilCuttingTransform", "img_w": int(tr["image_size"])},
            {"type": "RandomHorizontalFlip", "prob": float(aug.get("flip_prob", 0.2))},
            {"type": "RandomRotate", "prob": float(aug.get("rotate_prob", 0.2)), "degree": int(aug.get("rotate_degree", 8))},
            {"type": "RandomErasing", "prob": float(aug.get("erase_prob", 0.2))},
            {"type": "RandomPerspective", "prob": float(aug.get("perspective_prob", 0.1))},
        ],
    }]


def build_stage_cfg(base: Dict[str, Any], stage: Dict[str, Any], restore_ckpt: str, save_name: str, stage_iters: int) -> Dict[str, Any]:
    tr = base["training"]
    opengait_root = Path(base["project"]["opengait_root"]).expanduser().resolve()
    partition_abs = _resolve_partition_path(opengait_root, str(stage["dataset_partition"]))
    return {
        "data_cfg": {
            "dataset_name": stage["dataset_name"],
            "dataset_root": stage["dataset_root"],
            "dataset_partition": str(partition_abs),
            "num_workers": 4,
            "cache": False,
            "remove_no_gallery": False,
            "test_dataset_name": stage["dataset_name"],
        },
        "evaluator_cfg": {
            "enable_float16": True,
            "restore_ckpt_strict": False,
            "restore_hint": restore_ckpt,
            "save_name": save_name,
            "eval_func": stage.get("eval_func", "evaluate_indoor_dataset"),
            "sampler": {"batch_shuffle": False, "batch_size": 8, "sample_type": "all_ordered", "frames_all_limit": 720},
            "metric": "euc",
            "transform": [
                {"type": "ResizeSilTransform", "img_h": int(tr["image_size"]), "img_w": int(tr["image_size"]), "interpolation": "nearest"},
                {"type": "BaseSilCuttingTransform", "img_w": int(tr["image_size"])},
            ],
        },
        "loss_cfg": [
            {"loss_term_weight": 1.0, "margin": 0.2, "type": "TripletLoss", "log_prefix": "triplet"},
            {"loss_term_weight": 0.1, "scale": 16, "type": "CrossEntropyLoss", "log_prefix": "softmax", "log_accuracy": True},
        ],
        "model_cfg": {
            "model": "Baseline",
            "backbone_cfg": {"in_channels": 1, "layers_cfg": ["BC-64", "BC-64", "M", "BC-128", "BC-128", "M", "BC-256", "BC-256", "M", "BC-512", "BC-512"], "type": "Plain"},
            "SeparateFCs": {"in_channels": 512, "out_channels": 256, "parts_num": 31},
            "SeparateBNNecks": {"class_num": int(stage["class_num"]), "in_channels": 256, "parts_num": 31},
            "bin_num": [16, 8, 4, 2, 1],
        },
        "optimizer_cfg": {"solver": "SGD", "lr": float(tr["lr"]), "momentum": float(tr["momentum"]), "weight_decay": float(tr["weight_decay"])},
        "scheduler_cfg": {"scheduler": "MultiStepLR", "gamma": 0.1, "milestones": [int(stage_iters * 0.6), int(stage_iters * 0.85)]},
        "trainer_cfg": {
            "enable_float16": True,
            "fix_BN": bool(tr["fix_bn"]),
            "with_test": False,
            "log_iter": int(tr["log_iter"]),
            "restore_ckpt_strict": False,
            "restore_hint": restore_ckpt,
            "optimizer_reset": bool(tr["optimizer_reset"]),
            "scheduler_reset": bool(tr["scheduler_reset"]),
            "save_iter": int(min(tr.get("save_iter", stage_iters), stage_iters)),
            "save_name": save_name,
            "sync_BN": True,
            "total_iter": int(stage_iters),
            "sampler": {
                "type": "TripletSampler",
                "batch_shuffle": True,
                "batch_size": [int(tr["batch_p"]), int(tr["batch_k"])],
                "frames_num_fixed": int(tr["frames_num_fixed"]),
                "sample_type": str(tr["sample_type"]),
            },
            "transform": _train_transform_cfg(tr),
        },
    }


def expected_ckpt(opengait_root: Path, dataset_name: str, save_name: str, iteration: int) -> Path:
    return opengait_root / "output" / dataset_name / "Baseline" / save_name / "checkpoints" / f"{save_name}-{iteration:05d}.pt"


def _stage_ckpt_dir(opengait_root: Path, dataset_name: str, save_name: str) -> Path:
    return opengait_root / "output" / dataset_name / "Baseline" / save_name / "checkpoints"


def latest_stage_ckpt(opengait_root: Path, dataset_name: str, save_name: str) -> Path | None:
    ckpt_dir = _stage_ckpt_dir(opengait_root, dataset_name, save_name)
    if not ckpt_dir.is_dir():
        return None
    prefix = f"{save_name}-"
    candidates = []
    for p in ckpt_dir.glob(f"{save_name}-*.pt"):
        stem = p.stem
        if not stem.startswith(prefix):
            continue
        step_str = stem[len(prefix):]
        if not step_str.isdigit():
            continue
        candidates.append((int(step_str), p))
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])[1]


def run_cmd(cmd: List[str], cwd: Path, env: Dict[str, str], dry_run: bool) -> None:
    print(" ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--resume", action="store_true", help="Resume by skipping stages that already have checkpoints.")
    args = ap.parse_args()
    cfg_path = Path(args.config).expanduser().resolve()
    user_cfg = load_yaml(cfg_path)
    validate_user_paths(user_cfg)

    project = user_cfg["project"]
    stages = _enabled_stages_in_order(user_cfg["datasets"])
    opengait_root = Path(project["opengait_root"]).expanduser().resolve()
    work_dir = Path(project["work_dir"]).expanduser().resolve()
    gen_dir = work_dir / "generated_configs_aug"
    gen_dir.mkdir(parents=True, exist_ok=True)

    restore_ckpt = str(Path(user_cfg["checkpoint"]["pretrained_ckpt"]).expanduser().resolve())
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(project["gpus"])
    nproc = int(project["nproc_per_node"])
    dry_run = bool(project.get("dry_run", False))
    py = str(project.get("python", "python"))
    resume = bool(args.resume)

    for idx, stage in enumerate(stages, start=1):
        stage_iters = int(stage.get("stage_iters", 20000))
        stage_name = stage["name"]
        save_name = f"{project['name']}_stage{idx}_{stage_name.replace('-', '_')}"
        expected_final = expected_ckpt(opengait_root, stage["dataset_name"], save_name, stage_iters)
        latest_saved = latest_stage_ckpt(opengait_root, stage["dataset_name"], save_name)

        if resume and latest_saved is not None:
            use_ckpt = expected_final if expected_final.is_file() else latest_saved
            print(f"\n=== Stage {idx}: {stage_name} ===")
            print(f"Resume mode: found existing checkpoint, skipping retrain: {use_ckpt}")
            restore_ckpt = str(use_ckpt)
            continue

        cfg = build_stage_cfg(user_cfg, stage, restore_ckpt, save_name, stage_iters)
        cfg_path_stage = gen_dir / f"stage{idx}_{stage_name}.yaml"
        save_yaml(cfg_path_stage, cfg)
        print(f"\n=== Stage {idx}: {stage_name} ===")
        print(f"Config: {cfg_path_stage}")
        print(f"Init checkpoint: {restore_ckpt}")
        cmd = [py, "-m", "torch.distributed.launch", f"--nproc_per_node={nproc}", "opengait/main.py", "--cfgs", str(cfg_path_stage), "--phase", "train"]
        run_cmd(cmd, cwd=opengait_root, env=env, dry_run=dry_run)

        next_ck = expected_final
        if not dry_run and not next_ck.is_file():
            fallback = latest_stage_ckpt(opengait_root, stage["dataset_name"], save_name)
            if fallback is None:
                raise FileNotFoundError(f"Expected checkpoint missing after stage {idx}: {next_ck}")
            print(f"WARNING: expected final checkpoint not found, using latest saved checkpoint: {fallback}")
            next_ck = fallback
        restore_ckpt = str(next_ck)
    print(f"\nAll stages complete.\nFinal checkpoint: {restore_ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

