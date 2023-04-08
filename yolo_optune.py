import json
import os
from pathlib import Path

import numpy as np
import optuna
import torch
from optuna.study import StudyDirection

from configs import TEST_TRACKS_PATH, load_default_bound_line
from exception_tools import print_exception
from path_tools import get_video_files
from post_processing.alex import alex_count_humans
from post_processing.group_3 import group_3_count
from post_processing.timur import get_camera, timur_count_humans
from resultools import TestResults
from utils.general import set_logging
from yolo_track_bbox import YoloTrackBbox
from yolo_track_by_txt import run_track_yolo

print(f"optuna version = {optuna.__version__}")

cameras_info = {}


def run_track_yolo(txt_source_folder: str, source: str, tracker_type, tracker_config, reid_weights,
                   test_result_file, test_func=None, files=None,
                   classes=None, change_bb=None, conf=0.3):
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

    test_results = TestResults(test_result_file)

    # список файлов с видео для обработки
    list_of_videos = get_video_files(source, files)

    total_videos = len(list_of_videos)

    for i, item in enumerate(list_of_videos):
        print(f"process file: {i + 1}/{total_videos} {item}")

        run_single_video_yolo(txt_source_folder, item, tracker_type, tracker_config,
                              reid_weights, test_results, test_func, classes,
                              change_bb=change_bb,
                              conf=conf)
    # save results

    cmp_results = test_results.compare_list_to_file_v2(None, test_results.test_items)

    return cmp_results


def run_single_video_yolo(txt_source_folder, source, tracker_type: str, tracker_config,
                          reid_weights, test_file, test_func,
                          classes=None, change_bb=False, conf=0.3):
    # print(f"start {source}, {txt_source_folder}")

    source_path = Path(source)

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

    # track_worker = TrackWorker(track)

    num, w, h, fps = get_camera(source)
    bound_line = cameras_info.get(num)

    # print(f"num = {num}, w = {w}, h = {h}, bound_line = {bound_line}")

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
                if test_func == "group_3":
                    humans_result = group_3_count(tracks_new, num, w, h, fps)
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
            print_exception(e, "post processing")


def reset_seed(seed=123):
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def objective_osc(trial):
    """
      asso_func: giou
      conf_thres: 0.5122620708221085
      delta_t: 1
      det_thresh: 0
      inertia: 0.3941737016672115
      iou_thresh: 0.22136877277096445
      max_age: 50
      min_hits: 1
      use_byte: false

      ASSO_FUNCS = {  "iou": iou_batch,
                    "giou": giou_batch,
                    "ciou": ciou_batch,
                    "diou": diou_batch,
                    "ct_dist": ct_dist}

    """

    reset_seed()

    video_source = "d:\\AI\\2023\\corridors\\dataset-v1.1\\test\\"
    test_file = TEST_TRACKS_PATH
    reid_weights = "osnet_x0_25_msmt17.pt"

    files = None
    # files = ['1', "2", "3"]
    # files = ["3"]

    # classes = [0]
    classes = None

    change_bb = None  # pavel_change_bbox  # change_bbox

    # test_func = "popov_alex"
    test_func = "group_3"
    test_func = "timur"

    txt_source_folder = "D:\\AI\\2023\\Detect\\2023_03_29_10_35_01_YoloVersion.yolo_v7_detect"

    conf_thres = trial.suggest_float('conf_thres', 0.36, 0.56, log=True)
    max_age = int(trial.suggest_float('max_age', 1, 10, log=True))  #
    min_hits = int(trial.suggest_float('min_hits', 6, 8, log=True))  #
    iou_threshold = trial.suggest_float('iou_threshold', 0.62, 0.66, log=True)
    delta_t = int(trial.suggest_float('delta_t', 5, 8, log=True))  #
    asso_func = trial.suggest_categorical('asso', ["iou", "giou"])
    inertia = trial.suggest_float('inertia', 0.6, 0.75, log=True)
    use_byte = trial.suggest_categorical('use_byte', [True, False])

    # test_func = trial.suggest_categorical('test_func', ["timur", "group_3"])

    tracker_config = \
        {
            "det_thresh": conf_thres,
            "max_age": max_age,
            "min_hits": min_hits,
            "iou_thresh": iou_threshold,
            "delta_t": delta_t,
            "asso_func": asso_func,
            "inertia": inertia,
            "use_byte": use_byte
        }

    tracker_name = "ocsort"
    tracker_config = {tracker_name: tracker_config}

    cmp_results = run_track_yolo(txt_source_folder, video_source, tracker_name, tracker_config,
                                 reid_weights, test_file, test_func=test_func,
                                 files=files, change_bb=change_bb, classes=classes)

    acc = cmp_results["total_equal_percent"]

    return acc


def run_optuna():
    study = optuna.create_study(direction=StudyDirection.MAXIMIZE)
    study.optimize(objective_osc, n_trials=60)

    trial = study.best_trial

    print(trial)

    print('total_equal_percent: {}'.format(trial.value))
    print("Best hyper parameters: {}".format(trial.params))

    output_folder = "d:\\AI\\2023\\corridors\\dataset-v1.1\\"

    value_json_file = Path(output_folder) / f"ocsort_value.json"

    params_json_file = Path(output_folder) / f"ocsort_params.json"

    with open(value_json_file, "w") as write_file:
        write_file.write(json.dumps(trial.value, indent=4, sort_keys=True))

    with open(params_json_file, "w") as write_file:
        write_file.write(json.dumps(trial.params, indent=4, sort_keys=True))


if __name__ == '__main__':
    run_optuna()
