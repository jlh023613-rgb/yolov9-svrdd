"""复用 yolov8 的全部训练和评估参数，在 SVRDD 上训练 RT-DETR-L。."""

import yolov8

if __name__ == "__main__":
    yolov8.DATA = yolov8.ROOT / "datasets" / "SVRDD" / "data.yaml"
    # 项目会根据权重中的 RTDETR 解码头自动选择专用训练器和评估器。
    yolov8.MODEL = yolov8.ROOT / "rtdetr-l.pt"
    yolov8.main()
