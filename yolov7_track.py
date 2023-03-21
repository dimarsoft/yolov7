import json
import os
from pathlib import Path

from labeltools import TrackWorker
from post_processing.alex import alex_count_humans
from post_processing.timur import timur_count_humans, get_camera
from resultools import TestResults
from save_txt_tools import yolo7_save_tracks_to_txt
from utils.torch_utils import time_synchronized
from yolov7 import YOLO7
from datetime import datetime


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


def run_single_video_yolo7(model, source, tracker_type: str, tracker_config, output_folder,
                           reid_weights, test_file, test_func, conf=0.3, save_vid=False):
    print(f"start {source}")

    source_path = Path(source)
    text_path = Path(output_folder) / f"{source_path.stem}.txt"

    model = YOLO7(model)

    track = model.track(
        source=source,
        conf=conf,
        tracker_type=tracker_type,
        tracker_config=tracker_config,
        reid_weights=reid_weights
    )

    print(f"save tracks to: {text_path}")

    yolo7_save_tracks_to_txt(results=track, txt_path=text_path, conf=conf)

    track_worker = TrackWorker(track)

    if save_vid:
        t1 = time_synchronized()
        track_worker.create_video(source, output_folder)
        t2 = time_synchronized()

        print(f"Processed '{source}' to {output_folder}: ({(1E3 * (t2 - t1)):.1f} ms)")

    # count humans
    if test_func is None:
        # humans_result = track_worker.test_humans()
        # humans_result = alex_count_humans(track)
        tracks_new = []
        for item in track:
            tracks_new.append([item[0], item[5], item[6], item[1], item[2], item[3], item[4], item[7]])
        humans_result = timur_count_humans(tracks_new, source)
    else:
        #  info = [frame_id,
        #  left, top,
        #  width, height,
        #  int(detection[4]), int(detection[5]), float(detection[6])]
        # [frame_index, track_id, cls, bbox_left, bbox_top, bbox_w, bbox_h, box.conf]
        tracks_new = []
        for item in track:
            tracks_new.append([item[0], item[5], item[6], item[1], item[2], item[3], item[4], item[7]])
        humans_result = test_func(tracks_new)

    humans_result.file = source_path.name

    # add result
    test_file.add_test(humans_result)


def run_yolo7(model, source, tracker_type: str, tracker_config, output_folder, reid_weights,
              test_result_file, test_func=None, conf=0.3, save_vid=False):
    """

    Args:
        test_func: внешняя функция пользователя для постобработки
        test_result_file: эталонный файл разметки проходов людей
        reid_weights: веса для трекера, нукоторым нужны
        conf: conf для трекера
        save_vid: Создаем наше видео с центром человека
        output_folder: путь к папке для результатов работы, txt
        tracker_type: трекер (botsort.yaml, bytetrack.yaml)
        tracker_config: путь к своему файлу с настройками
        source: путь к видео, если папка, то для каждого видеофайла запустит
        model (object): модель для YOLO7
    """
    source_path = Path(source)

    # в выходной папке создаем папку с сессией: дата_трекер туда уже сохраняем все файлы

    now = datetime.now()

    session_folder_name = f"{now.year:04d}_{now.month:02d}_{now.day:02d}_{now.hour:02d}_{now.minute:02d}_" \
                          f"{now.second:02d}_y7_{tracker_type}"

    session_folder = str(Path(output_folder) / session_folder_name)

    try:
        os.makedirs(session_folder, exist_ok=True)
        print(f"Directory '{session_folder}' created successfully")
    except OSError as error:
        print(f"Directory '{session_folder}' can not be created")

    import shutil

    save_tracker_config = str(Path(session_folder) / Path(tracker_config).name)

    print(f"Copy '{tracker_config}' to '{save_tracker_config}")

    shutil.copy(tracker_config, save_tracker_config)

    save_test_result_file = str(Path(session_folder) / Path(test_result_file).name)

    print(f"Copy '{test_result_file}' to '{save_test_result_file}")

    shutil.copy(test_result_file, save_test_result_file)

    session_info = dict()

    session_info['model'] = str(Path(model).name)
    session_info['reid_weights'] = str(Path(reid_weights).name)
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
                run_single_video_yolo7(model, str(entry), tracker_type, tracker_config, session_folder,
                                       reid_weights, test_results, test_func, conf, save_vid)
    else:
        run_single_video_yolo7(model, source, tracker_type, tracker_config, session_folder,
                               reid_weights, test_results, test_func, conf, save_vid)

    # save results

    test_results.save_results(session_folder)

    test_results.compare_to_file_v2(session_folder)


def run_example():
    model = "D:\\AI\\2023\\models\\Yolov7\\25.02.2023_dataset_1.1_yolov7_best.pt"
    video_source = "d:\\AI\\2023\\corridors\\dataset-v1.1\\test\\1.mp4"
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

    tracker_config = "trackers/NorFairTracker/configs/norfair_track.yaml"
    run_yolo7(model, video_source, "norfair", tracker_config,
              output_folder, reid_weights, test_file, save_vid=True)


if __name__ == '__main__':
    # run_example()

    video_source = "d:\\AI\\2023\\corridors\\dataset-v1.1\\test\\20.mp4"

    num, w, h = get_camera(video_source)

    print(f"{num}, {w}, {h}")
