import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from configs import load_default_bound_line, CAMERAS_PATH, get_all_trackers_full_path, get_select_trackers, \
    TEST_TRACKS_PATH
from labeltools import TrackWorker
from path_tools import get_video_files
from post_processing.alex import alex_count_humans
from post_processing.group_3 import group_3_count
from post_processing.timur import timur_count_humans, get_camera
from resultools import TestResults, save_test_result, save_results_to_csv
from save_txt_tools import yolo7_save_tracks_to_txt, yolo7_save_tracks_to_json
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
    json_file = Path(output_folder) / f"{source_path.stem}.json"

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

    if save_vid:
        print(f"save tracks to: {text_path}")
        yolo7_save_tracks_to_txt(results=track, txt_path=text_path, conf=conf)

        yolo7_save_tracks_to_json(results=track, json_file=json_file, conf=conf)

    track_worker = TrackWorker(track)

    if save_vid:
        t1 = time_synchronized()
        track_worker.create_video(source, output_folder)
        t2 = time_synchronized()

        print(f"Processed '{source}' to {output_folder}: ({(1E3 * (t2 - t1)):.1f} ms)")

    num, w, h, fps = get_camera(source)
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
                if test_func == "group_3":
                    humans_result = group_3_count(tracks_new, num, w, h, fps)
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


def run_track_yolo(txt_source_folder: str, source: str, tracker_type, tracker_config, output_folder, reid_weights,
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

    if isinstance(tracker_type, dict):
        session_folder_name = f"{now.year:04d}_{now.month:02d}_{now.day:02d}_{now.hour:02d}_{now.minute:02d}_" \
                              f"{now.second:02d}_yolo_tracks_by_txt"
    else:
        session_folder_name = f"{now.year:04d}_{now.month:02d}_{now.day:02d}_{now.hour:02d}_{now.minute:02d}_" \
                              f"{now.second:02d}_yolo_tracks_by_txt_{tracker_type}"

    session_folder = str(Path(output_folder) / session_folder_name)

    try:
        os.makedirs(session_folder, exist_ok=True)
        print(f"Directory '{session_folder}' created successfully")
    except OSError as error:
        print(f"Directory '{session_folder}' can not be created. {error}")

    import shutil

    if not isinstance(tracker_type, dict):
        save_tracker_config = str(Path(session_folder) / Path(tracker_config).name)

        print(f"Copy '{tracker_config}' to '{save_tracker_config}")

        shutil.copy(tracker_config, save_tracker_config)

    save_test_result_file = str(Path(session_folder) / Path(test_result_file).name)

    print(f"Copy '{test_result_file}' to '{save_test_result_file}")

    shutil.copy(test_result_file, save_test_result_file)

    session_info = dict()

    session_info['txt_source_folder'] = str(txt_source_folder)
    session_info['reid_weights'] = str(Path(reid_weights).name)
    # session_info['conf'] = conf
    session_info['test_result_file'] = str(test_result_file)
    session_info['save_vid'] = save_vid
    session_info['files'] = files
    session_info['classes'] = classes
    session_info['change_bb'] = str(change_bb)
    session_info['cameras_path'] = str(CAMERAS_PATH)

    # test_tracks_file(test_result_file)

    if isinstance(test_func, str):
        session_info['test_func'] = test_func

    session_info_path = str(Path(session_folder) / 'session_info.json')

    with open(session_info_path, "w") as session_info_file:
        json.dump(session_info, fp=session_info_file, indent=4)

    test_results = TestResults(test_result_file)

    # список файлов с видео для обработки
    list_of_videos = get_video_files(source, files)

    total_videos = len(list_of_videos)

    if isinstance(tracker_type, dict):
        test_result_by_traker = {}
        for tracker in tracker_type.keys():
            tracker_config = tracker_type.get(tracker)
            tracker_session_folder = Path(session_folder) / str(tracker)

            try:
                os.makedirs(tracker_session_folder, exist_ok=True)
                print(f"Directory '{tracker_session_folder}' created successfully")

                save_tracker_config = str(Path(tracker_session_folder) / Path(tracker_config).name)

                print(f"Copy '{tracker_config}' to '{save_tracker_config}")

                shutil.copy(tracker_config, save_tracker_config)

            except OSError as error:
                print(f"Directory '{tracker_session_folder}' can not be created. {error}")

            test_results = TestResults(test_result_file)

            for i, item in enumerate(list_of_videos):
                print(f"process file: {i + 1}/{total_videos} {item}")

                run_single_video_yolo(txt_source_folder, item, tracker, tracker_config,
                                      tracker_session_folder,
                                      reid_weights, test_results, test_func, classes,
                                      change_bb=change_bb,
                                      conf=conf,
                                      save_vid=save_vid)

            file_result = save_test_result(test_results, tracker_session_folder, source_path)

            test_result_by_traker[str(tracker)] = file_result

            save_test_result_file = str(Path(tracker_session_folder) / Path(test_result_file).name)

            print(f"Copy '{test_result_file}' to '{save_test_result_file}")

            shutil.copy(test_result_file, save_test_result_file)

            # сохраняем результаты и в общий файл
            # и в каждой итерации файл обновляется новыми данными

            save_test_result_file = str(Path(session_folder) / 'all_compare_track_results.json')

            print(f"Save total result_items '{str(save_test_result_file)}'")
            with open(save_test_result_file, "w") as write_file:
                write_file.write(json.dumps(test_result_by_traker,
                                            indent=4, sort_keys=True, default=lambda o: o.__dict__))

            save_results_csv_file = str(Path(session_folder) / 'all_compare_track_results.csv')

            print(f"Save total to csv '{str(save_results_csv_file)}'")

            save_results_to_csv(test_result_by_traker, save_results_csv_file)
    else:
        for i, item in enumerate(list_of_videos):
            print(f"process file: {i + 1}/{total_videos} {item}")

            run_single_video_yolo(txt_source_folder, item, tracker_type, tracker_config,
                                  session_folder,
                                  reid_weights, test_results, test_func, classes,
                                  change_bb=change_bb,
                                  conf=conf,
                                  save_vid=save_vid)
        # save results

        save_test_result(test_results, session_folder, source_path)


def run_example():
    video_source = "d:\\AI\\2023\\corridors\\dataset-v1.1\\test\\"
    test_file = TEST_TRACKS_PATH
    output_folder = "d:\\AI\\2023\\corridors\\dataset-v1.1\\"
    reid_weights = "osnet_x0_25_msmt17.pt"

    # reid_weights = "D:\\AI\\2023\\Github\\dimar_yolov7\\weights\\mars-small128.pb"
    # tracker_config = "trackers/NorFairTracker/configs/norfair_track.yaml"

    all_trackers = get_all_trackers_full_path()

    # selected_trackers_names = ["fastdeepsort"]
    selected_trackers_names = ["ocsort"]

    selected_trackers = get_select_trackers(selected_trackers_names, all_trackers)

    tracker_name = selected_trackers  # "norfair"
    # tracker_name = selected_trackers  # "norfair"
    tracker_config = None  # all_trackers.get(tracker_name)

    files = None
    files = ['1']

    # classes = [0]
    classes = None

    change_bb = None  # pavel_change_bbox  # change_bbox

    # test_func = "popv_alex"
    # test_func = "group_3"
    test_func = "timur"

    txt_source_folder = "D:\\AI\\2023\\Detect\\2023_03_29_10_35_01_YoloVersion.yolo_v7_detect"
    run_track_yolo(txt_source_folder, video_source, tracker_name, tracker_config,
                   output_folder, reid_weights, test_file, test_func=test_func,
                   files=files, save_vid=False,  change_bb=change_bb, classes=classes)


# запуск из командной строки: python yolo_detect.py  --yolo 7 --weights "" source ""
def run_cli(opt_info):
    txt_source_folder, source, output_folder, files = \
        opt_info.txt_source_folder, opt_info.source, opt_info.output_folder, opt_info.files

    tracker_name, tracker_config = opt_info.tracker_name, opt_info.tracker_config
    test_file, test_func = opt_info.test_file, opt_info.test_func

    run_track_yolo(txt_source_folder, source, tracker_name, tracker_config, output_folder,
                   opt_info.reid_weights,
                   test_file, test_func=test_func, files=files,
                   conf=opt_info.conf, save_vid=opt_info.save_vid, classes=opt_info.classes)


if __name__ == '__main__':
    run_example()

    parser = argparse.ArgumentParser()
    parser.add_argument('--txt_source_folder', type=str, help='txt_source_folder')
    parser.add_argument('--source', type=str, help='source')  # file/folder, 0 for webcam
    parser.add_argument('--tracker_name', type=str, help='tracker_name')
    parser.add_argument('--tracker_config', type=str, help='tracker_config')
    parser.add_argument('--output_folder', type=str, help='output_folder')  # output folder
    parser.add_argument('--reid_weights', type=str, help='reid_weights')
    parser.add_argument('--test_file', type=str, help='test_file')
    parser.add_argument('--test_func', type=str, help='test_func')
    parser.add_argument('--files', type=list, default=None, help='files names list')  # files from list
    parser.add_argument('--classes', type=list, help='classes')
    parser.add_argument('--save_vid', type=bool, help='save results to *.mp4')
    parser.add_argument('--change_bb', default=None, help='change bbox, True, False, scale, function')
    parser.add_argument('--conf', type=float, default=0.3, help='object confidence threshold')
    opt = parser.parse_args()
    print(opt)

    #run_cli(opt)
