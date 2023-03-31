import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

from configs import load_default_bound_line, CAMERAS_PATH, get_all_trackers_full_path
from labeltools import TrackWorker
from post_processing.alex import alex_count_humans
from post_processing.timur import timur_count_humans, get_camera
from resultools import TestResults, test_tracks_file
from save_txt_tools import yolo7_save_tracks_to_txt
from utils.general import set_logging
from utils.torch_utils import time_synchronized
from yolo_track_bbox import YoloTrackBbox
from yolov7_track import save_exception

# настройки камер, считываются при старте сессии
cameras_info = {}


def run_single_video_yolo(txt_source_folder, source, tracker_type: str, tracker_config, output_folder,
                          reid_weights, test_file, test_func,
                          classes=None, change_bb=False, conf=0.3, save_vid=False):
    print(f"start {source}, {txt_source_folder}")

    source_path = Path(source)
    text_path = Path(output_folder) / f"{source_path.stem}.txt"

    txt_source = Path(txt_source_folder) / f"{source_path.stem}.txt"

    model = YoloTrackBbox()

    track = model.track(
        source=source,
        txt_source=txt_source,
        conf_threshold=conf,
        tracker_type=tracker_type,
        tracker_config=tracker_config,
        reid_weights=reid_weights,
        classes=classes,
        change_bb=change_bb
    )

    print(f"save tracks to: {text_path}")

    yolo7_save_tracks_to_txt(results=track, txt_path=text_path, conf=conf)

    track_worker = TrackWorker(track)

    if save_vid:
        t1 = time_synchronized()
        track_worker.create_video(source, output_folder)
        t2 = time_synchronized()

        print(f"Processed '{source}' to {output_folder}: ({(1E3 * (t2 - t1)):.1f} ms)")

    num, w, h = get_camera(source)
    bound_line = cameras_info.get(num)

    print(f"num = {num}, w = {w}, h = {h}, bound_line = {bound_line}")

    # count humans
    if test_func is not None:
        try:
            tracks_new = []
            for item in track:
                tracks_new.append([item[0], item[5], item[6], item[1], item[2], item[3], item[4], item[7]])

            if isinstance(test_func, str):

                humans_result = None

                if test_func == "popov_alex":
                    humans_result = alex_count_humans(tracks_new, num, w, h, bound_line)
                    pass
                if test_func == "timur":
                    humans_result = timur_count_humans(tracks_new, source)
                    pass
                if test_func == "dimar":
                    humans_result = track_worker.test_humans()
                    pass
                if humans_result is not None:
                    humans_result.file = source_path.name

                    # add result
                    test_file.add_test(humans_result)
            else:
                #  info = [frame_id,
                #  left, top,
                #  width, height,
                #  int(detection[4]), int(detection[5]), float(detection[6])]
                # [frame_index, track_id, cls, bbox_left, bbox_top, bbox_w, bbox_h, box.conf]
                # humans_result = test_func(tracks_new)
                # bound_line =  [[490, 662], [907, 613]]
                # num(str), w(int), h(int)

                humans_result = test_func(tracks_new, num, w, h, bound_line)
                humans_result.file = source_path.name
                # add result
                test_file.add_test(humans_result)

        except Exception as e:
            text_ex_path = Path(output_folder) / f"{source_path.stem}_pp_ex.log"
            save_exception(e, text_ex_path, "post processing")


def run_track_yolo(txt_source_folder: str, source: str, tracker_type: str, tracker_config, output_folder, reid_weights,
                   test_result_file, test_func=None, files=None,
                   classes=None, change_bb=None, conf=0.3, save_vid=False):
    """

    Args:
        txt_source_folder: папка с Labels: 1.txt....
        change_bb: менять bbox после детекции для трекера,
                   bbox будет меньше и по центру человека
                   если тип float, масштабируем по центру
                   если функция, то функция меняет ббокс
        files: если указана папка, но можно указать имена фай1лов,
                которые будут обрабатываться. ['1', '2' ...]
        classes: список классов, None все, [0, 1, 2....]
        test_func: внешняя функция пользователя для постобработки
        test_result_file: эталонный файл разметки проходов людей
        reid_weights: веса для трекера, некоторым нужны
        conf: conf для трекера
        save_vid: Создаем наше видео с центром человека
        output_folder: путь к папке для результатов работы, txt
        tracker_type: трекер (botsort, bytetrack)
        tracker_config: путь к своему файлу с настройками
        source: путь к видео, если папка, то для каждого видео файла запустит
    """

    set_logging()

    # при старте сессии считываем настройки камер
    global cameras_info
    cameras_info = load_default_bound_line()

    source_path = Path(source)

    # в выходной папке создаем папку с сессией: дата_трекер туда уже сохраняем все файлы

    now = datetime.now()

    session_folder_name = f"{now.year:04d}_{now.month:02d}_{now.day:02d}_{now.hour:02d}_{now.minute:02d}_" \
                          f"{now.second:02d}_yolo_tracks_by_txt_{tracker_type}"

    session_folder = str(Path(output_folder) / session_folder_name)

    try:
        os.makedirs(session_folder, exist_ok=True)
        print(f"Directory '{session_folder}' created successfully")
    except OSError as error:
        print(f"Directory '{session_folder}' can not be created. {error}")

    import shutil

    save_tracker_config = str(Path(session_folder) / Path(tracker_config).name)

    print(f"Copy '{tracker_config}' to '{save_tracker_config}")

    shutil.copy(tracker_config, save_tracker_config)

    save_test_result_file = str(Path(session_folder) / Path(test_result_file).name)

    print(f"Copy '{test_result_file}' to '{save_test_result_file}")

    shutil.copy(test_result_file, save_test_result_file)

    session_info = dict()

    session_info['txt_source_folder'] = str(txt_source_folder)
    session_info['reid_weights'] = str(Path(reid_weights).name)
    session_info['conf'] = conf
    session_info['test_result_file'] = test_result_file
    session_info['save_vid'] = save_vid
    session_info['files'] = files
    session_info['classes'] = classes
    session_info['change_bb'] = str(change_bb)
    session_info['cameras_path'] = str(CAMERAS_PATH)

    test_tracks_file(test_result_file)

    if isinstance(test_func, str):
        session_info['test_func'] = test_func

    session_info_path = str(Path(session_folder) / 'session_info.json')

    with open(session_info_path, "w") as session_info_file:
        json.dump(session_info, fp=session_info_file, indent=4)

    test_results = TestResults(test_result_file)

    if source_path.is_dir():
        print(f"process folder: {source_path}")

        for entry in source_path.iterdir():
            # check if it is a file
            if entry.is_file() and entry.suffix == ".mp4":
                if files is None:
                    run_single_video_yolo(txt_source_folder, str(entry), tracker_type, tracker_config,
                                          session_folder,
                                          reid_weights, test_results, test_func,
                                          classes, change_bb, conf, save_vid)
                else:
                    if entry.stem in files:
                        run_single_video_yolo(txt_source_folder, str(entry), tracker_type, tracker_config,
                                              session_folder,
                                              reid_weights, test_results, test_func,
                                              classes, change_bb, conf, save_vid)

    else:
        print(f"process file: {source_path}")
        run_single_video_yolo(txt_source_folder, source, tracker_type, tracker_config,
                              session_folder,
                              reid_weights, test_results, test_func, classes,
                              change_bb=change_bb,
                              conf=conf,
                              save_vid=save_vid)

    # save results

    try:
        test_results.save_results(session_folder)
    except Exception as e:
        text_ex_path = Path(session_folder) / f"{source_path.stem}_ex_result.log"
        with open(text_ex_path, "w") as write_file:
            write_file.write("Exception in save_results!!!")
            exc_type, exc_value, exc_traceback = sys.exc_info()
            lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            for line in lines:
                write_file.write(line)
            for item in test_results.result_items:
                write_file.write(f"{str(item)}\n")

        print(f"Exception in save_results {str(e)}! details in {str(text_ex_path)} ")

    try:
        test_results.compare_to_file_v2(session_folder)
    except Exception as e:
        text_ex_path = Path(session_folder) / f"{source_path.stem}_ex_compare.log"
        save_exception(e, text_ex_path, "compare_to_file_v2")


def change_bbox(bbox, file_id: str):
    """

    Args:
        file_id(str): имя файла
        bbox (tensor): первые 4ре столбца x1 y1 x2 y2
        в абсолютных значениях картинки.
        Пока это ббоксы всех классов,
        сам класс находится в последнем столбце (-1, или индекс 5)
    """
    print(f"file = {file_id}")
    return bbox


# пример реализации бокса по центру 20/20
def change_bbox_to_center(bbox, file_id):
    x1 = (bbox[:, [0]] + bbox[:, [2]]) / 2
    y1 = (bbox[:, [1]] + bbox[:, [3]]) / 2

    w = 10
    h = 10

    bbox[:, [0]] = x1 - w
    bbox[:, [2]] = x1 + w

    bbox[:, [1]] = y1 - h
    bbox[:, [3]] = y1 + h

    return bbox


# пример реализации от Павла (группа №1)
def pavel_change_bbox(bbox, file_id):
    y2 = bbox[:, [1]] + 150

    bbox[:, [3]] = y2

    return bbox


def run_example():
    video_source = "d:\\AI\\2023\\corridors\\dataset-v1.1\\test\\"
    test_file = "D:\\AI\\2023\\TestInfo\\all_track_results.json"
    output_folder = "d:\\AI\\2023\\corridors\\dataset-v1.1\\"
    reid_weights = "osnet_x0_25_msmt17.pt"
    reid_weights = "D:\\AI\\2023\\Github\\dimar_yolov7\\weights\\mars-small128.pb"

    # tracker_config = "trackers/NorFairTracker/configs/norfair_track.yaml"

    all_trackers = get_all_trackers_full_path()

    tracker_name = "deepsort"
    tracker_config = all_trackers.get(tracker_name)

    files = None
    files = ['1']

    change_bb = False # change_bbox

    txt_source_folder = "D:\\AI\\2023\\Detect\\2023_03_29_10_35_01_YoloVersion.yolo_v7_detect"
    run_track_yolo(txt_source_folder, video_source, tracker_name, tracker_config,
                   output_folder, reid_weights, test_file, test_func="timur",
                   files=files, save_vid=True, change_bb=change_bb)


if __name__ == '__main__':
    run_example()
