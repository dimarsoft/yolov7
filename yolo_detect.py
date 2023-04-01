import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from configs import parse_yolo_version, YoloVersion
from labeltools import TrackWorker
from save_txt_tools import yolo7_save_tracks_to_txt, yolo7_save_tracks_to_json
from utils.general import set_logging
from utils.torch_utils import time_synchronized
from yolov7 import YOLO7
from yolov8 import YOLO8
from yolov8_ultralitics import YOLO8UL


def create_yolo_model(yolo_version, model):
    if yolo_version == YoloVersion.yolo_v7:
        return YOLO7(model)

    if yolo_version == YoloVersion.yolo_v8:
        return YOLO8(model)

    if yolo_version == YoloVersion.yolo_v8ul:
        return YOLO8UL(model)


def detect_single_video_yolo(yolo_version, model, source, output_folder, classes=None, conf=0.1, save_txt=True,
                             save_vid=False):
    print(f"start detect_single_video_yolo: {yolo_version}, source = {source}")

    source_path = Path(source)

    model = create_yolo_model(yolo_version, model)

    detections = model.detect(
        source=source,
        conf_threshold=conf,
        classes=classes
    )

    if save_txt:
        text_path = Path(output_folder) / f"{source_path.stem}.txt"

        print(f"save detections to: {text_path}")

        yolo7_save_tracks_to_txt(results=detections, txt_path=text_path, conf=conf)

        json_file = Path(output_folder) / f"{source_path.stem}.json"

        print(f"save detections to: {json_file}")

        yolo7_save_tracks_to_json(results=detections, json_file=json_file, conf=conf)

    if save_vid:
        track_worker = TrackWorker(detections)
        t1 = time_synchronized()
        track_worker.create_video(source, output_folder, draw_class=True)
        t2 = time_synchronized()

        print(f"Processed '{source}' to {output_folder}: ({(1E3 * (t2 - t1)):.1f} ms)")


def run_detect_yolo(yolo_info, model: str, source: str, output_folder,
                    files=None, classes=None, conf=0.3, save_txt=True, save_vid=False):
    """

    Args:
        yolo_info: версия Yolo: 7 ил 8
        save_txt: сохранять бб в файл
        files: если указана папка, но можно указать имена фай1лов,
                которые будут обрабатываться. ['1', '2' ...]
        classes: список классов, None все, [0, 1, 2....]
        conf: conf
        save_vid: Создаем видео c bb
        output_folder: путь к папке для результатов работы, txt
        source: путь к видео, если папка, то для каждого видео файла запустит
        model (str): модель для YOLO8
    """

    set_logging()

    print(f"yolo version = {yolo_info}")
    yolo_version = parse_yolo_version(yolo_info)

    if yolo_version is None:
        raise Exception(f"unsupported yolo version {yolo_info}")

    source_path = Path(source)

    # в выходной папке создаем папку с сессией: дата_трекер туда уже сохраняем все файлы

    now = datetime.now()

    session_folder_name = f"{now.year:04d}_{now.month:02d}_{now.day:02d}_{now.hour:02d}_{now.minute:02d}_" \
                          f"{now.second:02d}_{yolo_version}_detect"

    session_folder = str(Path(output_folder) / session_folder_name)

    try:
        os.makedirs(session_folder, exist_ok=True)
        print(f"Directory '{session_folder}' created successfully")
    except OSError as error:
        print(f"Directory '{session_folder}' can not be created. {error}")

    session_info = dict()

    session_info['model'] = str(Path(model).name)
    session_info['conf'] = conf
    session_info['save_vid'] = save_vid
    session_info['files'] = files
    session_info['classes'] = classes
    session_info['save_txt'] = save_txt
    session_info['yolo_version'] = str(yolo_version)

    session_info_path = str(Path(session_folder) / 'session_info.json')

    with open(session_info_path, "w") as session_info_file:
        json.dump(session_info, fp=session_info_file, indent=4)

    # test_results = TestResults(test_result_file)

    if source_path.is_dir():
        print(f"process folder: {source_path}")

        for entry in source_path.iterdir():
            # check if it is a file
            if entry.is_file() and entry.suffix == ".mp4":
                if files is None:
                    detect_single_video_yolo(yolo_version, model, str(entry), session_folder,
                                             classes, conf, save_txt, save_vid)
                else:
                    if entry.stem in files:
                        detect_single_video_yolo(yolo_version, model, str(entry), session_folder,
                                                 classes, conf, save_txt, save_vid)
    else:
        print(f"process file: {source_path}")
        detect_single_video_yolo(yolo_version, model, str(source_path), session_folder,
                                 classes, conf, save_txt, save_vid)


def run_example():
    video_source = "d:\\AI\\2023\\corridors\\dataset-v1.1\\test\\"
    output_folder = "d:\\AI\\2023\\corridors\\dataset-v1.1\\"

    files = ['1']

    #  model = "D:\\AI\\2023\\models\\Yolov7\\25.02.2023_dataset_1.1_yolov7_best.pt"

    #  run_detect_yolo(7, model, video_source, output_folder, files=files, conf=0.25, save_txt=True, save_vid=True)

    model = "D:\\AI\\2023\\models\\Yolo8s_batch32_epoch100.pt"

    run_detect_yolo(8, model, video_source, output_folder, files=files, conf=0.25, save_txt=True, save_vid=True)


# запуск из командной строки: python yolo_detect.py  --yolo 7 --weights "" source ""
def run_cli(opt_info):
    yolo, source, weights, output_folder, files, save_txt, save_vid, conf, classes = \
        opt_info.yolo, opt_info.source, opt_info.weights, opt_info.output_folder, \
        opt_info.files, opt_info.save_txt, opt_info.save_vid, opt_info.conf, opt_info.classes

    run_detect_yolo(yolo, weights, source, output_folder,
                    files=files, conf=conf, save_txt=save_txt, save_vid=save_vid, classes=classes)


if __name__ == '__main__':
    # run_example()

    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo', type=int, help='7, 8, 8ul')
    parser.add_argument('--weights', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, help='source')  # file/folder, 0 for webcam
    parser.add_argument('--files', type=str, default=None, help='files names list')  # files from list
    parser.add_argument('--output_folder', type=str, help='output_folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save_txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save_vid', action='store_true', help='save results to *.mp4')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    print(opt)

    run_cli(opt)
