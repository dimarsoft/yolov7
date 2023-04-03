import time
from pathlib import Path

import cv2
import numpy as np
import torch

from change_bboxes import change_bbox
from utils.general import check_img_size, xyxy2xywh

from configs import WEIGHTS
from models.experimental import attempt_load
from trackers.multi_tracker_zoo import create_tracker
from utils.datasets import letterbox, LoadImages
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device, time_synchronized, TracedModel
from yolo_tools import xyxy2ltwh


class YOLO7:
    def __init__(self, weights_path, half=False, device='', imgsz=640):
        self.device = select_device(device)
        # Load model
        self.weights_path = weights_path
        self.model = attempt_load(weights_path, map_location=self.device)  # load FP32 model

        self.stride = int(self.model.stride.max())  # model stride

        self.imgsz = check_img_size(imgsz, s=self.stride)  # check img_size

        self.model = TracedModel(self.model, self.device, img_size=self.imgsz)

        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        self.half = half
        if half:
            self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        print(f"device = {self.device}, half = {self.half}")
        if self.half:
            self.model.half()  # to FP16

        self.reid_weights = Path(WEIGHTS) / 'osnet_x0_25_msmt17.pt'  # model.pt path,

        self.augment = False
        self.agnostic_nms = False

        print(f"augment = {self.augment}, agnostic_nms = {self.agnostic_nms}")

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

    def detect2(self, source, conf_threshold=0.25, iou=0.45, classes=None):
        dataset = LoadImages(source, img_size=self.imgsz, stride=self.stride)

        results = []

        t0 = time.time()
        total_detections = 0

        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if self.device.type != 'cpu' and (
                    old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    self.model(img, augment=self.augment)[0]

            # Inference
            t1 = time_synchronized()
            with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
                pred = self.model(img, augment=self.augment)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, conf_threshold, iou, classes=classes, agnostic=self.agnostic_nms)

            t3 = time_synchronized()

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in det:
                        total_detections += 1

                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        results.append([frame, -1, cls, xywh[0], xywh[1], xywh[2], xywh[3], conf])

                # Print time (inference + NMS)
                print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS, detections = {total_detections}')

                # Save results (image with detections)

        print(f"total detections = {total_detections}")
        print(f'Done. ({time.time() - t0:.3f}s)')

        return results

    def detect(self, source, conf_threshold=0.25, iou=0.45, classes=None):

        input_video = cv2.VideoCapture(source)

        fps = int(input_video.get(cv2.CAP_PROP_FPS))
        # ширина
        w = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        # высота
        h = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # количество кадров в видео
        frames_in_video = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"input = {source}, w = {w}, h = {h}, fps = {fps}, frames_in_video = {frames_in_video}")
        results = []

        for frame_id in range(frames_in_video):

            ret, frame = input_video.read()

            # Inference
            t1 = time_synchronized()
            s = ""

            with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
                new_frame = self.to_tensor(frame)
                predict = self.model(new_frame, augment=self.augment)[0]

            t2 = time_synchronized()

            # Apply NMS
            predict = non_max_suppression(predict, conf_threshold, iou, classes=classes, agnostic=self.agnostic_nms)
            t3 = time_synchronized()

            dets = 0
            empty_conf_count = 0

            gn = torch.tensor(frame.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            for tr_id, det in enumerate(predict):
                if len(det) > 0:

                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(new_frame.shape[2:], det[:, :4], frame.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    for *xyxy, conf, cls in det:
                        dets += 1

                        # normalized ltwh
                        ltwh = (xyxy2ltwh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()

                        left = ltwh[0]
                        top = ltwh[1]
                        width = ltwh[2]
                        height = ltwh[3]

                        info = [frame_id,
                                left, top,
                                width, height,
                                # id
                                int(-1),
                                # cls
                                int(cls),
                                # conf
                                float(conf)]

                        results.append(info)

            t4 = time_synchronized()

            detections_info = f"{s}{'' if dets > 0 else ', (no detections)'}"

            empty_conf_count_str = f"{'' if empty_conf_count == 0 else f', empty_confs = {empty_conf_count}'}"

            # Print time (inference + NMS)

            print(f'frame ({frame_id + 1}/{frames_in_video}) Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, '
                  f'({(1E3 * (t3 - t2)):.1f}ms) NMS, {(1E3 * (t4 - t3)):.1f}ms) '
                  f'{detections_info} {empty_conf_count_str}, {len(results)}')

        input_video.release()

        print(f"Done. total detections = {len(results)}")

        return results

    def track(self, source, tracker_type, tracker_config, reid_weights="osnet_x0_25_msmt17.pt",
              conf_threshold=0.3, iou=0.4, classes=None, change_bb=False):

        self.reid_weights = Path(WEIGHTS) / reid_weights
        tracker = create_tracker(tracker_type, tracker_config, self.reid_weights, self.device, self.half)

        file_id = Path(source).stem

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

        results, results_det = [], []

        for frame_id in range(frames_in_video):
            ret, frame = input_video.read()

            # Inference
            t1 = time_synchronized()
            s = ""

            with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
                new_frame = self.to_tensor(frame)
                predict = self.model(new_frame)[0]
                t2 = time_synchronized()

                # Apply NMS
                predict = non_max_suppression(predict, conf_threshold, iou, classes=classes)
                t3 = time_synchronized()

                curr_frame = frame

                if hasattr(tracker, 'camera_update'):
                    if prev_frame is not None and curr_frame is not None:  # camera motion compensation
                        tracker.camera_update(prev_frame, curr_frame)

                dets = 0
                empty_conf_count = 0

                for tr_id, predict_track in enumerate(predict):
                    if predict_track is not None and len(predict_track) > 0:
                        dets += 1
                        predict_track = change_bbox(predict_track, change_bb, file_id)

                        # conf_ = predict_track[:, [4]]
                        # cls = predict_track[:, [5]]

                        # print(f"cls = {cls}")
                        # print(f"conf_ = {conf_}")

                        # Rescale boxes from img_size to im0 size
                        conv_pred = scale_coords(new_frame.shape[2:], predict_track, frame.shape).round()

                        # Print results
                        for c in predict_track[:, 5].unique():
                            n = (predict_track[:, 5] == c).sum()  # detections per class
                            s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        for det_id, detection in enumerate(predict_track):  # detections per image
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

                            conf = detection[4]
                            cls = detection[5]

                            if conf is None:
                                # print("detection[6] is None")
                                empty_conf_count += 1
                                continue

                            info = [frame_id,
                                    left, top,
                                    width, height,
                                    # id
                                    int(-1),
                                    # cls
                                    int(cls),
                                    # conf
                                    float(conf)]

                            # print(info)
                            results_det.append(info)

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

                            # print(info)
                            results.append(info)

            t4 = time_synchronized()

            detections_info = f"{s}{'' if dets > 0 else ', (no detections)'}"

            empty_conf_count_str = f"{'' if empty_conf_count == 0 else f', empty_confs = {empty_conf_count}'}"

            prev_frame = frame

            # Print time (inference + NMS)

            print(f'frame ({frame_id + 1}/{frames_in_video}) Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, '
                  f'({(1E3 * (t3 - t2)):.1f}ms) NMS, {(1E3 * (t4 - t3)):.1f}ms) '
                  f'{detections_info} {empty_conf_count_str}')

        input_video.release()

        return results, results_det

    def train(self, **kwargs):
        pass