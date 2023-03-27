from ultralytics import YOLO
from pathlib import Path

from ultralytics.yolo.data.augment import LetterBox

from configs import WEIGHTS
from save_txt_tools import convert_toy7
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
        detections = self.model.predict(source, conf=conf, iou=iou, classes=classes)

        detections = convert_toy7(detections.cpy(), save_none_id=True)

        return detections
