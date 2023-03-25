import json
import os
from datetime import datetime

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path

from ultralytics.nn.autobackend import AutoBackend
from ultralytics.yolo.data.augment import LetterBox
from ultralytics.yolo.utils.ops import scale_boxes, non_max_suppression

from labeltools import TrackWorker
from resultools import TestResults
from save_txt_tools import yolo8_save_tracks_to_txt, convert_toy7
from trackers.multi_tracker_zoo import create_tracker
from utils.torch_utils import time_synchronized, select_device
from yolov7 import YOLO7

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'


class YOLO8:
    def __init__(self, weights_path, half=False, device=''):
        self.device = select_device(device)

        self.half = half
        if half:
            self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        print(f"device = {self.device}, half = {self.half}")

        # Load model

        self.model = AutoBackend(weights=weights_path,
                                 fp16=self.half,
                                 device=self.device)

        self.names = self.model.names

        #        if self.half:
        #            self.model.half()  # to FP16

        self.reid_weights = Path(WEIGHTS) / 'osnet_x0_25_msmt17.pt'  # model.pt path,

        self.letter_box = LetterBox()

    def to_tensor(self, img):

        # Padded resize
        img = self.letter_box(image=img)

        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)  # contiguous

        # Convert
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        return img

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
            s = ""

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

                dets = 0
                empty_conf_count = 0
                for tr_id, predict_track in enumerate(predict):
                    if predict_track is not None and len(predict_track) > 0:

                        predict_track = YOLO7.change_bbox(predict_track, change_bb)

                        dets += 1
                        # bbox = predict_track[:, :4]
                        # conf_ = predict_track[:, [4]]
                        # cls = predict_track[:, [5]]

                        # print(f"cls = {cls}")
                        # print(f"conf_ = {conf_}")
                        # print(f"bbox = {bbox}")

                        # Rescale boxes from img_size to im0 size
                        predict_track[:, :4] = scale_boxes(new_frame.shape[2:], predict_track[:, :4],
                                                           frame.shape).round()  # rescale boxes to im0 size

                        # Print results
                        for c in predict_track[:, 5].unique():
                            n = (predict_track[:, 5] == c).sum()  # detections per class
                            s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        # conv_pred = scale_coords(new_frame.shape[2:], predict_track, frame.shape).round()

                        tracker_outputs = tracker.update(predict_track.cpu(), frame)

                        # print(f"tracker_outputs count = {len(tracker_outputs)}")

                        # Process detections [f, x1, y1, x2, y2, track_id, class_id, conf]

                        # empty_conf_count += (tracker_outputs[:, 6] is None).sum()  # detections per class

                        for det_id, detection in enumerate(tracker_outputs):  # detections per image

                            # bbox = detection[0:4]
                            # track_id = detection[4]
                            # cls = detection[5]
                            # conf = detection[6]

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

                detections_info = f"{s}{'' if dets > 0 else ', (no detections)'}"

                empty_conf_count_str = f"{'' if empty_conf_count == 0 else f', empty_confs = {empty_conf_count}'}"

                t4 = time_synchronized()

                prev_frame = frame

                # Print total time (preprocessing + inference + NMS + tracking)

                # Print time (inference + NMS)
                print(f'frame ({frame_id + 1}/{frames_in_video}) Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, '
                      f'({(1E3 * (t3 - t2)):.1f}ms) NMS, {(1E3 * (t4 - t3)):.1f}ms) '
                      f'{detections_info} {empty_conf_count_str}')

        input_video.release()

        return results


def run_single_video_yolo8(model, source, tracker, output_folder, test_file, test_func,
                           conf=0.3, save_vid=False, save_vid2=False):
    print(f"start {source}")
    model = YOLO(model)

    track = model.track(
        source=source,
        stream=False,
        save=save_vid,
        conf=conf,
        tracker=tracker
    )
    source_path = Path(source)
    text_path = Path(output_folder) / f"{source_path.stem}.txt"

    print(f"save to: {text_path}")

    yolo8_save_tracks_to_txt(results=track, txt_path=text_path, conf=conf)

    tracks_y7 = convert_toy7(track)
    track_worker = TrackWorker(tracks_y7)

    if save_vid2:
        t1 = time_synchronized()
        track_worker.create_video(source, output_folder)
        t2 = time_synchronized()

        print(f"Processed '{source}' to {output_folder}: ({(1E3 * (t2 - t1)):.1f} ms)")

    # count humans
    if test_func is None:
        humans_result = track_worker.test_humans()
    else:
        #  info = [frame_id,
        #  left, top,
        #  width, height,
        #  int(detection[4]), int(detection[5]), float(detection[6])]
        # [frame_index, track_id, cls, bbox_left, bbox_top, bbox_w, bbox_h, box.conf]
        tracks_new = []
        for item in tracks_y7:
            tracks_new.append([item[0], item[5], item[6], item[1], item[2], item[3], item[4], item[7]])

        humans_result = test_func(tracks_new)

    humans_result.file = source_path.name

    # add result
    test_file.add_test(humans_result)


def run_yolo8(model: str, source, tracker, output_folder, test_result_file, test_func,
              conf=0.3, save_vid=False, save_vid2=False):
    """

    Args:
        test_func: def count(tracks) -> Result
        test_result_file: файл с разметкой для проверки
        conf: conf для трекера
        save_vid2: Создаем наше видео с центром человека
        save_vid (Bool): save для model.track. Yolo создает свое видео
        output_folder: путь к папке для результатов работы, txt
        tracker: трекер (botsort.yaml, bytetrack.yaml) или путь к своему файлу с настройками
        source: путь к видео, если папка, то для каждого видеофайла запустит
        model (object): модель для YOLO8
    """
    source_path = Path(source)

    # в выходной папке создаем папку с сессией: дата_трекер туда уже сохраняем все файлы

    now = datetime.now()

    # tracker_path = Path(tracker)

    session_folder_name = f"{now.year:04d}_{now.month:02d}_{now.day:02d}_{now.hour:02d}_{now.minute:02d}_" \
                          f"{now.second:02d}_y8_{tracker}"

    session_folder = str(Path(output_folder) / session_folder_name)

    try:
        os.makedirs(session_folder, exist_ok=True)
        print(f"Directory '{session_folder}' created successfully")
    except OSError as error:
        print(f"Directory '{session_folder}' can not be created. {error}")

    import shutil

    # save_tracker_config = str(Path(session_folder) / tracker_path.name)

    # print(f"Copy '{tracker_path}' to '{save_tracker_config}")

    # shutil.copy(str(tracker_path), save_tracker_config)

    save_test_result_file = str(Path(session_folder) / Path(test_result_file).name)

    print(f"Copy '{test_result_file}' to '{save_test_result_file}")

    shutil.copy(test_result_file, save_test_result_file)

    # заполняем информацию о сессии
    session_info = dict()

    session_info['model'] = str(Path(model).name)
    session_info['conf'] = conf
    session_info['test_result_file'] = test_result_file

    session_info_path = str(Path(session_folder) / 'session_info.json')

    print(f"Save session to '{session_info_path}")

    with open(session_info_path, "w") as session_info_file:
        json.dump(session_info, fp=session_info_file, indent=4)

    test_results = TestResults(test_result_file)

    if source_path.is_dir():
        print(f"process folder: {source_path}")

        for entry in source_path.iterdir():
            # check if it is a file
            if entry.is_file() and entry.suffix == ".mp4":
                run_single_video_yolo8(model, str(entry), tracker, session_folder,
                                       test_results, test_func, conf, save_vid, save_vid2)

        # test_results.compare_to_file(session_folder)
    else:
        run_single_video_yolo8(model, source, tracker, session_folder,
                               test_results, test_func, conf, save_vid, save_vid2)
        # test_results.compare_one_to_file(session_folder)

    # save results

    test_results.save_results(session_folder)
    test_results.compare_to_file_v2(session_folder)


def run_example():
    model = "D:\\AI\\2023\\models\\Yolo8s_batch32_epoch100.pt"
    video_source = "d:\\AI\\2023\\corridors\\dataset-v1.1\\test\\"
    test_file = "D:\\AI\\2023\\TestInfo\\all_track_results.json"

    tracker_config = "./trackers/strongsort/configs/strongsort.yaml"
    output_folder = "d:\\AI\\2023\\corridors\\dataset-v1.1\\"
    reid_weights = "osnet_x0_25_msmt17.pt"
    # run_yolo7(model, video_source, "strongsort", tracker_config, output_folder, reid_weights, test_file)

    tracker_config = "trackers/deep_sort/configs/deepsort.yaml"
    reid_weights = "mars-small128.pb"
    # run_yolo7(model, video_source, "deepsort", tracker_config, output_folder, reid_weights)

    tracker_config = "trackers/botsort/configs/botsort.yaml"
    reid_weights = "osnet_x0_25_msmt17.pt"
    # run_yolo7(model, video_source, "botsort", tracker_config, output_folder, reid_weights)

    tracker_config = "trackers/ocsort/configs/ocsort.yaml"
    reid_weights = "osnet_x0_25_msmt17.pt"
    # run_yolo7(model, video_source, "ocsort", tracker_config, output_folder, reid_weights)

    tracker_config = "trackers/bytetrack/configs/bytetrack.yaml"
    reid_weights = "osnet_x0_25_msmt17.pt"
    # run_yolo7(model, video_source, "bytetrack", tracker_config, output_folder, reid_weights)

    tracker_config = "trackers/fast_deep_sort/configs/fastdeepsort.yaml"
    reid_weights = "mars-small128.pb"
    # run_yolo7(model, video_source, "fastdeepsort", tracker_config,
    #          output_folder, reid_weights, test_file, save_vid=True)

    tracker_config = "trackers/botsort/configs/botsort.yaml"
    run_yolo8(model, video_source, "botsort",
              output_folder, test_file, test_func=None)


if __name__ == '__main__':
    run_example()
