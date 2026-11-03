import csv
import hashlib
import json
import platform
import shutil
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

import ultralytics
from ultralytics import YOLO
from ultralytics.data.utils import IMG_FORMATS
from ultralytics.utils.torch_utils import get_flops, get_num_params

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "yolov8s.pt"  # 每次从原始预训练权重开始；更换模型时也保持下面的训练协议一致。
DATA = ROOT / "datasets" / "RDD2022" / "data.yaml"
TRAIN_SETTINGS = dict(
    imgsz=1024,  # 保持已有对比实验的输入尺寸；RDD 图片会按此尺寸缩放和填充。
    batch=16,  # 对比实验保持一致，并核对结果中记录的实际 batch。
    device=0,
    workers=8,
    epochs=500,
    patience=40,  # 保留早停：验证集 mAP50-95 连续 40 轮未提升就停止。
    seed=42,
    deterministic=True,
    optimizer="SGD",
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    nbs=64,  # 名义累积 batch；实际梯度累积次数也记录在结果中。
    warmup_epochs=3.0,
    cos_lr=False,
    close_mosaic=10,
    amp=True,
    cache=False,
    fraction=1.0,
    val=True,
    plots=True,
    save=True,
    save_period=-1,  # 保存 best.pt、last.pt，不额外保存每轮权重。
)


def snapshot_experiment(trainer):
    """在训练开始时保存实验来源，避免中途停止后缺失配置和数据划分记录。"""
    run_dir = Path(trainer.save_dir)
    pretrained_weights = Path(trainer.args.model)
    shutil.copy2(__file__, run_dir / "script_snapshot.py")
    shutil.copy2(DATA, run_dir / "data_snapshot.yaml")
    splits, manifests = {}, {}
    for split in ("train", "val", "test"):
        if not trainer.data.get(split):
            continue
        image_dir = Path(trainer.data[split])
        images = sorted(p for p in image_dir.rglob("*") if p.suffix.lower()[1:] in IMG_FORMATS)
        manifests[split] = [p.relative_to(DATA.parent).as_posix() for p in images]
        digest = hashlib.sha256()
        counts = [0] * len(trainer.data["names"])
        for image, relative in zip(images, manifests[split]):
            label = image_dir.parent / "labels" / image.relative_to(image_dir).with_suffix(".txt")
            content = label.read_bytes()
            digest.update(relative.encode("utf-8") + b"\0" + content + b"\0")
            for line in content.decode("utf-8").splitlines():
                counts[int(line.split()[0])] += 1
        splits[split] = dict(images=len(images), instances_per_class=counts, paths_and_labels_sha256=digest.hexdigest())
    (run_dir / "split_images.json").write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    report = {
        "pretrained_weights": str(pretrained_weights),
        "pretrained_sha256": hashlib.sha256(pretrained_weights.read_bytes()).hexdigest(),
        "dataset": {"config": str(DATA), "names": trainer.data["names"], "splits": splits},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "ultralytics": ultralytics.__version__,
            "ultralytics_source": str(Path(ultralytics.__file__).resolve()),
            "torch": torch.__version__,
            "packages": {name: version(name) for name in ("torchvision", "numpy", "opencv-python", "polars")},
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(trainer.device),
            "gpu_memory_gib": torch.cuda.get_device_properties(trainer.device).total_memory / 2**30,
        },
        "protocol": "根据 val 选择 best.pt 和 baseline；仅评估 data.yaml 配置的有标注划分，未配置的 test 不读取。",
        "results": {},
    }
    (run_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    model_path = Path(MODEL)
    settings = TRAIN_SETTINGS.copy()
    name = (
        f"{DATA.parent.name.lower()}_{model_path.stem}_img{settings['imgsz']}"
        f"_b{settings['batch']}_seed{settings['seed']}"
    )
    model = YOLO(str(model_path))
    model.add_callback("on_train_start", snapshot_experiment)
    start = time.perf_counter()
    model.train(
        **settings,
        data=str(DATA),
        project=str(ROOT / "runs" / "detect"),
        name=name,
        exist_ok=False,
    )

    # 使用本次 trainer 的实际目录和 best.pt，避免误用自动递增前的旧实验。
    trainer = model.trainer
    run_dir, best_weights = Path(trainer.save_dir), Path(trainer.best)
    report = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    report["best_weights"] = str(best_weights)
    report["training"] = {
        "epochs_completed": trainer.epoch + 1,
        "best_epoch": trainer.stopper.best_epoch,  # 从 1 开始，按训练期间的 val mAP50-95 选择。
        "actual_train_batch": trainer.batch_size,
        "gradient_accumulation": trainer.accumulate,
        "wall_seconds_including_setup_and_final_val": time.perf_counter() - start,
        "peak_allocated_gpu_gib": torch.cuda.max_memory_allocated(settings["device"]) / 2**30,
        "amp_enabled": trainer.amp,
    }
    report["model"] = {
        "parameters": get_num_params(trainer.model),
        "gflops_at_imgsz": get_flops(trainer.model, settings["imgsz"]) or None,
        "best_weights_mib": best_weights.stat().st_size / 2**20,
    }
    report["evaluation_settings"] = {
        **{k: settings[k] for k in ("imgsz", "batch", "device", "workers", "seed", "deterministic", "plots")},
        "data": str(DATA),
        "conf": 0.001,  # 计算 AP 的低阈值；不是部署时的置信度阈值。
        "iou": 0.7,
        "max_det": 300,
        "quantize": 32,
        "augment": False,
        "rect": False,
        "save_txt": True,
        "save_conf": True,
        "save_json": False,  # 保存 YOLO txt，沿用原生 AP 协议，无需额外 COCO 评估依赖。
        "verbose": True,
    }
    (run_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    model = YOLO(str(best_weights))
    comparison = []
    for split in (("val", "test") if trainer.data.get("test") else ("val",)):
        metrics = model.val(
            **report["evaluation_settings"], split=split, project=str(run_dir), name=split, exist_ok=False
        )
        output = Path(metrics.save_dir)
        (output / "per_class.csv").write_text(metrics.to_csv(decimals=8), encoding="utf-8-sig")
        (output / "confusion_matrix.csv").write_text(metrics.confusion_matrix.to_csv(), encoding="utf-8-sig")
        np.savez_compressed(
            output / "curves.npz",
            **{
                k: getattr(metrics.box, k)
                for k in ("ap_class_index", "all_ap", "px", "p_curve", "r_curve", "f1_curve", "prec_values")
            },
        )
        scores = {
            **metrics.results_dict,
            "metrics/mAP75(B)": float(metrics.box.map75),
            "metrics/mean_F1(B)": float(np.mean(metrics.box.f1)) if len(metrics.box.f1) else 0.0,
        }
        report["results"][split] = {"metrics": scores, "speed_ms_per_image": metrics.speed, "save_dir": str(output)}
        (run_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        comparison.append(
            {
                "model": model_path.stem,
                "split": split,
                "seed": settings["seed"],
                "imgsz": settings["imgsz"],
                "requested_batch": settings["batch"],
                **report["training"],
                **report["model"],
                **scores,
                **{f"{k}_ms_per_image": v for k, v in metrics.speed.items()},
            }
        )
        with (run_dir / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=comparison[0])
            writer.writeheader()
            writer.writerows(comparison)
        print(f"\n{split}: mAP50={metrics.box.map50:.4f}, mAP50-95={metrics.box.map:.4f}，结果：{output}")

    print(f"\n实验全部结果：{run_dir}")


if __name__ == "__main__":
    main()
