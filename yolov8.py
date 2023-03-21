import json
import os
from datetime import datetime

from ultralytics import YOLO
from pathlib import Path

from labeltools import TrackWorker
from resultools import TestResults
from save_txt_tools import yolo8_save_tracks_to_txt, convert_toy7
from utils.torch_utils import time_synchronized


def create_video_with_track(results, source_video, output_file):
    import cv2

    input_video = cv2.VideoCapture(source_video)

    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    # ширина
    w = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    # высота
    h = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # количесто кадров в видео
    frames_in_video = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"input = {source_video}, w = {w}, h = {h}, fps = {fps}, frames_in_video = {frames_in_video}")

    output_video = cv2.VideoWriter(str(output_file), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    # считываем все фреймы из видео
    for i in range(frames_in_video):
        ret, frame = input_video.read()

        output_video.write(frame)

    output_video.release()
    input_video.release()


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

    tracker_path = Path(tracker)

    session_folder_name = f"{now.year:04d}_{now.month:02d}_{now.day:02d}_{now.hour:02d}_{now.minute:02d}_" \
                          f"{now.second:02d}_y8_{tracker_path.stem}"

    session_folder = str(Path(output_folder) / session_folder_name)

    try:
        os.makedirs(session_folder, exist_ok=True)
        print(f"Directory '{session_folder}' created successfully")
    except OSError as error:
        print(f"Directory '{session_folder}' can not be created")

    import shutil

    save_tracker_config = str(Path(session_folder) / tracker_path.name)

    print(f"Copy '{tracker_path}' to '{save_tracker_config}")

    shutil.copy(str(tracker_path), save_tracker_config)

    save_test_result_file = str(Path(session_folder) / Path(test_result_file).name)

    print(f"Copy '{test_result_file}' to '{save_test_result_file}")

    shutil.copy(test_result_file, save_test_result_file)

    session_info = dict()

    session_info['model'] = str(Path(model).name)
    # session_info['reid_weights'] = str(Path(reid_weights).name)
    session_info['conf'] = conf
    session_info['test_result_file'] = test_result_file

    session_info_path = str(Path(session_folder) / 'session_info.json')

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

