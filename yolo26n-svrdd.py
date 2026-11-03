"""复用 yolov8 的全部训练和评估参数，在 SVRDD 上训练 YOLO26n。"""

import yolov8


if __name__ == "__main__":
    yolov8.DATA = yolov8.ROOT / "datasets" / "SVRDD" / "data.yaml"
    yolov8.MODEL = yolov8.ROOT / "yolo26n.pt"
    yolov8.main()
