"""复用 yolov8 的训练和评估流程，保留已有 YOLO11s 的 SVRDD 实验设置。."""

import yolov8

if __name__ == "__main__":
    yolov8.DATA = yolov8.ROOT / "datasets" / "SVRDD" / "data.yaml"
    yolov8.MODEL = yolov8.ROOT / "yolo11s.pt"
    yolov8.main()
