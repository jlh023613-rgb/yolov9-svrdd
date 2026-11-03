"""临时冒烟测试：验证 YOLOv9t 能在 SVRDD 上走通 yolo8n-svrdd.py 的同一条训练路径。"""

from pathlib import Path

from ultralytics import YOLO

import yolov8

yolov8.DATA = yolov8.ROOT / "datasets" / "SVRDD" / "data.yaml"
yolov8.MODEL = yolov8.ROOT / "yolov9t.pt"

model = YOLO(str(yolov8.MODEL))
model.add_callback("on_train_start", yolov8.snapshot_experiment)
model.train(
    data=str(yolov8.DATA),
    imgsz=640,
    batch=4,
    epochs=1,
    fraction=0.01,
    device=0,
    workers=2,
    plots=False,
    val=False,
    save=False,
    verbose=False,
    project=str(yolov8.ROOT / "runs" / "detect"),
    name="smoke_yolov9t",
    exist_ok=True,
)
run_dir = Path(model.trainer.save_dir)
print("SMOKE_OK", run_dir, (run_dir / "metrics.json").exists())
