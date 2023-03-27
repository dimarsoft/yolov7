from ultralytics import YOLO
from pathlib import Path

from ultralytics.yolo.data.augment import LetterBox

from configs import WEIGHTS
from utils.torch_utils import select_device


class YOLO8UL:
    def __init__(self, weights_path, half=False, device=''):
        self.device = select_device(device)

        self.half = half
        if half:
            self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        print(f"device = {self.device}, half = {self.half}")

        self.model = YOLO(weights_path)

        self.names = self.model.names

        #        if self.half:
        #            self.model.half()  # to FP16

        self.reid_weights = Path(WEIGHTS) / 'osnet_x0_25_msmt17.pt'  # model.pt path,

        self.letter_box = LetterBox()

    def detect(self, source, conf=0.3, iou=0.4, classes=None):
        return self.model.predict(source, conf=conf, iou=iou, classes=classes)
