from pathlib import Path

import optuna

from configs import TEST_TRACKS_PATH, load_default_bound_line
from exception_tools import print_exception
from path_tools import get_video_files
from post_processing.alex import alex_count_humans
from post_processing.group_3 import group_3_count
from post_processing.timur import get_camera, timur_count_humans
from resultools import TestResults
from utils.general import set_logging
from yolo_track_bbox import YoloTrackBbox

print(f"optuna version = {optuna.__version__}")

cameras_info = {}


def run_track_yolo(txt_source_folder: str, source: str, tracker_type, tracker_config,
                   reid_weights="osnet_x0_25_msmt17.pt",
                   test_result_file=TEST_TRACKS_PATH, test_func=None, files=None,
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
        change_bb=change_bb,
        log=False
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
