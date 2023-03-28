from pathlib import Path

import cv2
import torch
from ultralytics.yolo.data.augment import LetterBox

from configs import WEIGHTS
from save_txt_tools import yolo_load_detections_from_txt
from trackers.multi_tracker_zoo import create_tracker
from utils.torch_utils import select_device, time_synchronized
from yolov7 import YOLO7


class YoloTrackBbox:

    def __init__(self, half=False, device=''):
        self.device = select_device(device)

        self.half = half
        if half:
            self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        print(f"device = {self.device}, half = {self.half}")

        self.reid_weights = ""

        self.letter_box = LetterBox()

    def det_to_tensor(self, df, w, h):
        res = []
        for item in df.values:
            ww = item[5] * w
            hh = item[6] * h
            x1 = item[3] * w
            y1 = item[4] * h
            conf = item[7]
            clss = item[2]
            res.append([x1, y1, ww + x1, hh + y1, conf, clss])

        return torch.tensor(res, device=self.device)

    def track(self, source, txt_source, tracker_type, tracker_config, reid_weights="osnet_x0_25_msmt17.pt",
              conf_threshold=0.3,
              iou=0.4,
              classes=None, change_bb=False):

        self.reid_weights = Path(WEIGHTS) / reid_weights
        tracker = create_tracker(tracker_type, tracker_config, self.reid_weights, self.device, self.half)

        need_camera_update = hasattr(tracker, 'camera_update')

        df_bbox = yolo_load_detections_from_txt(txt_source)
        img_frames = df_bbox[0].unique()

        input_video = cv2.VideoCapture(source)

        fps = int(input_video.get(cv2.CAP_PROP_FPS))
        # ширина
        w = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        # высота
        h = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # количество кадров в видео
        frames_in_video = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"input = {source}, w = {w}, h = {h}, fps = {fps}, frames_in_video = {frames_in_video}")

        curr_frame, prev_frame = None, None

        results = []

        for frame_id in img_frames:  # range(frames_in_video):

            input_video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = input_video.read()

            curr_frame = frame

            if need_camera_update:
                if prev_frame is not None and curr_frame is not None:  # camera motion compensation
                    tracker.camera_update(prev_frame, curr_frame)

            df_bbox_det = df_bbox[df_bbox[0] == frame_id]

            t1 = time_synchronized()

            with torch.no_grad():  # Calculating gradients would cause a GPU memory leak

                group = df_bbox_det[df_bbox_det[0] == frame_id]

                predict = self.det_to_tensor(group, w, h)

                predict = YOLO7.change_bbox(predict, change_bb)

                dets = 0
                empty_conf_count = 0

                tracker_outputs = tracker.update(predict.cpu(), frame)
                for det_id, detection in enumerate(tracker_outputs):  # detections per image

                    dets += 1

                    x1 = float(detection[0]) / w
                    y1 = float(detection[1]) / h
                    x2 = float(detection[2]) / w
                    y2 = float(detection[3]) / h

                    left = min(x1, x2)
                    top = min(y1, y2)
                    width = abs(x1 - x2)
                    height = abs(y1 - y2)

                    if detection[6] is None:
                        # print("detection[6] is None")
                        empty_conf_count += 1
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

                    results.append(info)

            t2 = time_synchronized()

            detections_info = f"({dets} tracks)"

            empty_conf_count_str = f"{'' if empty_conf_count == 0 else f', empty_confs = {empty_conf_count}'}"

            prev_frame = frame

            print(f'frame ({frame_id + 1}/{frames_in_video}) Done. ({(1E3 * (t2 - t1)):.1f}ms) tracking, '
                  f'{detections_info} {empty_conf_count_str}')

        input_video.release()

        return results
