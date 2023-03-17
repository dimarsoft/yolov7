from pathlib import Path

import cv2
import numpy as np
import torch

from models.experimental import attempt_load
from trackers.multi_tracker_zoo import create_tracker
from utils.general import non_max_suppression
from utils.torch_utils import select_device, time_synchronized

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'


class YOLO7:
    def __init__(self, weights_path, device=''):
        self.device = select_device(device)
        # Load model
        self.model = attempt_load(weights_path, map_location=self.device)  # load FP32 model

        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        if self.half:
            self.model.half()  # to FP16

        self.reid_weights = Path(WEIGHTS) / 'osnet_x0_25_msmt17.pt'  # model.pt path,

    def to_tensor(self, frame):
        img = frame  # , _, _ = letterbox(frame)

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img

    def track(self, source, tracker_type, tracker_config, reid_weights="osnet_x0_25_msmt17.pt", conf=0.3, iou=0.4,
              classes=None):

        tracker = create_tracker(tracker_type, tracker_config, self.reid_weights, self.device, self.half)

        input_video = cv2.VideoCapture(source)

        fps = int(input_video.get(cv2.CAP_PROP_FPS))
        # ширина
        w = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        # высота
        h = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # количесто кадров в видео
        frames_in_video = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"input = {source}, w = {w}, h = {h}, fps = {fps}, frames_in_video = {frames_in_video}")

        curr_frame, prev_frame = None, None

        for frame_id in range(frames_in_video):
            ret, frame = input_video.read()

            # Inference
            t1 = time_synchronized()

            with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
                predict = self.model(self.to_tensor(frame))[0]
                t2 = time_synchronized()

                # Apply NMS
                predict = non_max_suppression(predict, conf, iou, classes=classes)
                t3 = time_synchronized()

                curr_frame = frame

                if hasattr(tracker, 'camera_update'):
                    if prev_frame is not None and curr_frame is not None:  # camera motion compensation
                        tracker.camera_update(prev_frame, curr_frame)

                for tr_id, predict_track in enumerate(predict):
                    tracker_outputs = tracker.update(predict_track.cpu(), frame)

                    print(f"predict_track = {len(tracker_outputs)}")

                    # Process detections [x1, y1, x2, y2, track_id, class_id, conf, queue]
                    for det_id, detection in enumerate(tracker_outputs):  # detections per image
                        print(f"{det_id}: bb = {detection[:4]}, id = {detection[4]}, cls = {detection[5]}, conf = {detection[6]}")

            t4 = time_synchronized()

            prev_frame = frame

            # Print time (inference + NMS)
            print(f'frame ({frame_id + 1}/{frames_in_video}) Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, '
                  f'({(1E3 * (t3 - t2)):.1f}ms) NMS, {(1E3 * (t4 - t3)):.1f}ms) track')

        input_video.release()