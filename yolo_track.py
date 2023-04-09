import argparse
import json
import os
import shutil
from pathlib import Path

from change_bboxes import pavel_change_bbox
from configs import parse_yolo_version, get_all_trackers_full_path, get_select_trackers, TEST_TRACKS_PATH
from exception_tools import save_exception
from labeltools import TrackWorker
from path_tools import create_session_folder, get_video_files
from post_processing.alex import alex_count_humans
from post_processing.timur import get_camera, timur_count_humans
from resultools import TestResults, save_test_result, save_results_to_csv
from save_txt_tools import yolo7_save_tracks_to_txt
from utils.general import set_logging
from utils.torch_utils import time_synchronized
from yolo_detect import create_yolo_model
from yolo_track_by_txt import cameras_info


def run_single_video_yolo(yolo_version, model, source, tracker_type, tracker_config, output_folder,
                          reid_weights, test_file, test_func,
                          classes=None, change_bb=False, conf=0.3, iou=0.45, save_txt=False, save_vid=False):
    print(f"start detect_single_video_yolo: {yolo_version}, source = {source}")

    source_path = Path(source)

    text_path = Path(output_folder) / f"{source_path.stem}.txt"

    model = create_yolo_model(yolo_version, model)

    track = model.track(
        source=source,
        conf_threshold=conf,
        iou=iou,
        tracker_type=tracker_type,
        tracker_config=tracker_config,
        reid_weights=reid_weights,
        classes=classes,
        change_bb=change_bb
    )

    print(f"save tracks to: {text_path}")

    if save_txt:
        yolo7_save_tracks_to_txt(results=track, txt_path=text_path, conf=conf)

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


def run_track_yolo(yolo_info, model: str, source: str,
                   tracker_type: str, tracker_config,
                   output_folder, reid_weights,
                   test_result_file, test_func=None,
                   files=None, classes=None, change_bb=None,
                   conf=0.3, iou=0.45, save_vid=False, save_txt=False):
    """

    Args:
        yolo_info: версия Yolo: 7, 8 или 8ul
        model (str): модель для YOLO детекции
        source: путь к видео, если папка, то для каждого видео файла запустит
        tracker_type: Название трекера или словарь, если их несколько
        tracker_config: путь к файлу настройки трекера
        output_folder: путь к папке для результатов работы, txt
        reid_weights: все модели трекера
        test_result_file: файл для проверки результатов постобработки
        test_func: функция постобработки
        files: если указана папка, но можно указать имена фай1лов,
                которые будут обрабатываться. ['1', '2' ...]
        classes: список классов для детекции, None все, [0, 1, 2....]
        change_bb: Функция изменения бб
        conf: порог conf для детекции
        iou:
        save_txt: сохранять бб в файл
        save_vid: Создаем видео c bb
    """

    set_logging()

    print(f"yolo version = {yolo_info}")
    yolo_version = parse_yolo_version(yolo_info)

    if yolo_version is None:
        raise Exception(f"unsupported yolo version {yolo_info}")

    source_path = Path(source)

    # в выходной папке создаем папку с сессией: дата_трекер туда уже сохраняем все файлы

    session_folder = create_session_folder(yolo_version, output_folder, "track")

    session_info = dict()

    session_info['model'] = str(Path(model).name)
    session_info['conf'] = conf
    session_info['iou'] = iou
    session_info['save_vid'] = save_vid
    session_info['files'] = files
    session_info['classes'] = classes
    session_info['save_txt'] = save_txt
    session_info['yolo_version'] = str(yolo_version)

    session_info_path = str(Path(session_folder) / 'session_info.json')

    with open(session_info_path, "w") as session_info_file:
        json.dump(session_info, fp=session_info_file, indent=4)

    test_results = TestResults(test_result_file)

    # список файлов с видео для обработки
    list_of_videos = get_video_files(source_path, files)

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

                run_single_video_yolo(yolo_version, model, str(item),
                                      tracker_type=tracker, tracker_config=tracker_config,
                                      output_folder=tracker_session_folder,
                                      test_file=test_result_file, test_func=test_func,
                                      reid_weights=reid_weights,
                                      classes=classes,
                                      change_bb=change_bb,
                                      conf=conf, iou=iou,
                                      save_txt=save_txt, save_vid=save_vid)

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
            save_results_excel_file = str(Path(session_folder) / 'all_compare_track_results.xlsx')

            print(f"Save total to csv '{str(save_results_csv_file)}'")

            save_results_to_csv(test_result_by_traker, save_results_csv_file, save_results_excel_file)
    else:
        for i, item in enumerate(list_of_videos):
            print(f"process file: {i + 1}/{total_videos} {item}")

            run_single_video_yolo(yolo_version, model, str(item),
                                  tracker_type=tracker_type, tracker_config=tracker_config,
                                  output_folder=session_folder,
                                  test_file=test_result_file, test_func=test_func,
                                  reid_weights=reid_weights,
                                  classes=classes,
                                  change_bb=change_bb,
                                  conf=conf, iou=iou,
                                  save_txt=save_txt, save_vid=save_vid)
        # save results

        save_test_result(test_results, session_folder, source_path)


def run_example():
    video_source = "d:\\AI\\2023\\corridors\\dataset-v1.1\\test\\"
    output_folder = "d:\\AI\\2023\\corridors\\dataset-v1.1\\"

    files = ['1']

    video_source = "d:\\AI\\2023\\corridors\\dataset-v1.1\\test\\"
    test_file = TEST_TRACKS_PATH
    output_folder = "d:\\AI\\2023\\corridors\\dataset-v1.1\\"
    reid_weights = "osnet_x0_25_msmt17.pt"

    # reid_weights = "D:\\AI\\2023\\Github\\dimar_yolov7\\weights\\mars-small128.pb"

    # tracker_config = "trackers/NorFairTracker/configs/norfair_track.yaml"

    all_trackers = get_all_trackers_full_path()

    selected_trackers_names = ["ocsort"]  # "sort",

    selected_trackers = get_select_trackers(selected_trackers_names, all_trackers)

    tracker_name = all_trackers  # selected_trackers  # "norfair"
    # tracker_name = selected_trackers  # "norfair"
    tracker_config = None  # all_trackers.get(tracker_name)

    files = None
    files = ['1']

    classes = [0]
    classes = None

    change_bb = pavel_change_bbox  # change_bbox

    yolo7_w = "D:\\AI\\2023\\models\\Yolov7\\25.02.2023_dataset_1.1_yolov7_best.pt"
    yolo8_w = "D:\\AI\\2023\\models\\Yolo8s_batch32_epoch100.pt"

    test_func = None

    run_track_yolo("8ul",
                   yolo8_w, video_source,
                   tracker_type=tracker_name,
                   tracker_config=tracker_config,
                   output_folder=output_folder,
                   reid_weights=reid_weights,
                   test_result_file=test_file,
                   test_func=test_func,
                   change_bb=change_bb,
                   files=files, conf=0.4, save_txt=True, save_vid=True, classes=classes)


# запуск из командной строки: python yolo_detect.py  --yolo 7 --weights "" source ""
def run_cli(opt_info):
    yolo, source, weights, output_folder, files, save_txt, save_vid, conf, classes = \
        opt_info.yolo, opt_info.source, opt_info.weights, opt_info.output_folder, \
        opt_info.files, opt_info.save_txt, opt_info.save_vid, opt_info.conf, opt_info.classes

    tracker_name, tracker_config = opt_info.tracker_name, opt_info.tracker_config
    test_file, test_func = opt_info.test_file, opt_info.test_func

    run_track_yolo(yolo, weights, source,
                   tracker_type=tracker_name,
                   tracker_config=tracker_config,
                   output_folder=output_folder,
                   reid_weights=opt_info.reid_weights,
                   test_result_file=test_file,
                   test_func=test_func,
                   change_bb=opt_info.change_bb,
                   files=files, conf=conf, iou=opt_info.iou,
                   save_txt=save_txt, save_vid=save_vid, classes=classes)


if __name__ == '__main__':
    # run_example()

    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo', type=int, help='7, 8, 8ul')
    parser.add_argument('--weights', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--reid_weights', type=str, help='reid_weights')
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
