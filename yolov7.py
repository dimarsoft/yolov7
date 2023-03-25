from pathlib import Path

import cv2
import numpy as np
import torch

from models.experimental import attempt_load
from trackers.multi_tracker_zoo import create_tracker
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device, time_synchronized

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'


class YOLO7:
    def __init__(self, weights_path, half=False, device=''):
        self.device = select_device(device)
        # Load model
        self.model = attempt_load(weights_path, map_location=self.device)  # load FP32 model

        self.half = half
        if half:
            self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        print(f"device = {self.device}, half = {self.half}")
        if self.half:
            self.model.half()  # to FP16

        self.reid_weights = Path(WEIGHTS) / 'osnet_x0_25_msmt17.pt'  # model.pt path,

    def to_tensor(self, frame):
        img = frame  # , _, _ = letterbox(frame)

        # Padded resize
        img = letterbox(img)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        # img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img

    def to_tensor2(self, im):
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        return im

    @staticmethod
    def change_bbox(bbox, change_bb):

        if change_bb is None:
            return bbox

        if isinstance(change_bb, float):
            return YOLO7.change_bbox_v2(bbox, change_bb)

        if not isinstance(change_bb, bool):
            return bbox

        if not change_bb:
            return bbox

        x1 = (bbox[:, [0]] + bbox[:, [2]]) / 2
        y1 = (bbox[:, [1]] + bbox[:, [3]]) / 2

        w = 10  # abs(bbox[:, [0]] - bbox[:, [2]]) / 4
        h = 10  # abs(bbox[:, [1]] - bbox[:, [3]]) / 4

        bbox[:, [0]] = x1 - w
        bbox[:, [2]] = x1 + w

        bbox[:, [1]] = y1 - h
        bbox[:, [3]] = y1 + h

        return bbox

    @staticmethod
    def change_bbox_v2(bbox, scale: float):
        """

        Args:
            bbox: bbox на замену
            scale (float):
        """
        x1_center = (bbox[:, [0]] + bbox[:, [2]]) / 2
        y1_center = (bbox[:, [1]] + bbox[:, [3]]) / 2

        scale /= 2

        w = abs(bbox[:, [0]] - bbox[:, [2]]) * scale
        h = abs(bbox[:, [1]] - bbox[:, [3]]) * scale

        bbox[:, [0]] = x1_center - w
        bbox[:, [2]] = x1_center + w

        bbox[:, [1]] = y1_center - h
        bbox[:, [3]] = y1_center + h

        return bbox

    def track(self, source, tracker_type, tracker_config, reid_weights="osnet_x0_25_msmt17.pt", conf=0.3, iou=0.4,
              classes=None, change_bb=False):

        self.reid_weights = Path(WEIGHTS) / reid_weights
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

        results = []

        for frame_id in range(frames_in_video):
            ret, frame = input_video.read()

            # Inference
            t1 = time_synchronized()

            with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
                new_frame = self.to_tensor(frame)
                predict = self.model(new_frame)[0]
                t2 = time_synchronized()

                # Apply NMS
                predict = non_max_suppression(predict, conf, iou, classes=classes)
                t3 = time_synchronized()

                curr_frame = frame

                if hasattr(tracker, 'camera_update'):
                    if prev_frame is not None and curr_frame is not None:  # camera motion compensation
                        tracker.camera_update(prev_frame, curr_frame)

                for tr_id, predict_track in enumerate(predict):

                    predict_track = self.change_bbox(predict_track, change_bb)

                    # conf_ = predict_track[:, [4]]
                    # cls = predict_track[:, [5]]

                    # print(f"cls = {cls}")
                    # print(f"conf_ = {conf_}")

                    # Rescale boxes from img_size to im0 size
                    conv_pred = scale_coords(new_frame.shape[2:], predict_track, frame.shape).round()

                    tracker_outputs = tracker.update(conv_pred.cpu(), frame)
                    # tracker_outputs = tracker.update(predict_track.cpu(), frame)

                    # print(f"predict_track = {len(tracker_outputs)}")

                    # Process detections [f, x1, y1, x2, y2, track_id, class_id, conf]
                    for det_id, detection in enumerate(tracker_outputs):  # detections per image
                        # print(f"{det_id}: detection = {detection}")
                        # print(f"{det_id}: bb = {detection[:4]}, id = {detection[4]}, cls = {detection[5]}, "
                        #      f"conf = {detection[6]}")

                        x1 = float(detection[0]) / w
                        y1 = float(detection[1]) / h
                        x2 = float(detection[2]) / w
                        y2 = float(detection[3]) / h

                        left = min(x1, x2)
                        top = min(y1, y2)
                        width = abs(x1 - x2)
                        height = abs(y1 - y2)

                        if detection[6] is None:
                            print("detection[6] is None")
                            continue

                        info = [frame_id,
                                left, top,
                                width, height,
                                # id
                                int(detection[4]),
                                # cls
                                int(detection[5]),
                                # conf
                                float(detection[6])]

                        # print(info)
                        results.append(info)

            t4 = time_synchronized()

            prev_frame = frame

            # Print time (inference + NMS)
            print(f'frame ({frame_id + 1}/{frames_in_video}) Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, '
                  f'({(1E3 * (t3 - t2)):.1f}ms) NMS, {(1E3 * (t4 - t3)):.1f}ms) track')

        input_video.release()

        return results
