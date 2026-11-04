"""复用 yolov8.py 的全部训练和评估参数，在 SVRDD 上训练 YOLOv9t-LSK。."""

import yolov8  # 训练/评估协议只在 yolov8.py 维护一份，这里指定数据、模型结构和预训练权重。

if __name__ == "__main__":
    yolov8.DATA = (yolov8.ROOT / "datasets" / "SVRDD" / "data.yaml").resolve()
    yolov8.MODEL = yolov8.ROOT / "ultralytics" / "cfg" / "models" / "v9" / "yolov9t-lsk.yaml"
    yolov8.TRAIN_SETTINGS["pretrained"] = str(yolov8.ROOT / "yolov9t.pt")
    yolov8.main()
