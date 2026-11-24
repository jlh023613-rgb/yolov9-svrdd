"""复用 yolov8.py 的全部训练和评估参数，在 SVRDD 上训练 YOLOv11n."""

import yolov8  # 训练/评估协议只在 yolov8.py 维护一份，这里仅替换数据和权重。

if __name__ == "__main__":
    yolov8.DATA = yolov8.ROOT / "datasets" / "SVRDD" / "data.yaml"
    yolov8.MODEL = yolov8.ROOT / "yolo11n.pt"
    yolov8.main()
