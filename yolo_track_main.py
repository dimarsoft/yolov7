"""

"""
import json
from pathlib import Path
import gdown

from configs import load_default_bound_line, get_all_trackers_full_path, WEIGHTS, YoloVersion, parse_yolo_version, ROOT, \
    get_all_optune_trackers, TEST_TRACKS_PATH
from count_results import Result
from exception_tools import print_exception
from post_processing.alex import alex_count_humans
from post_processing.timur import get_camera, timur_count_humans
from resultools import results_to_json, TestResults
from yolo_detect import create_yolo_model

folder_link = "https://drive.google.com/drive/folders/1b-tp_yxHgadeElP4XoDCFoXxCwXHK9CV"
yolo7_model_gdrive = "https://drive.google.com/drive/u/4/folders/1b-tp_yxHgadeElP4XoDCFoXxCwXHK9CV"
yolo7_model_gdrive_file = "25.02.2023_dataset_1.1_yolov7_best.pt"
yolo8_model_gdrive_file = "640img_8x_best_b16_e10.pt"

test_video_share_folder_link = "https://drive.google.com/drive/folders/1YK0a3peuwdbvoZUAKciCvYM5KjKeizA6?usp=sharing"


def get_local_path(yolo_version: YoloVersion) -> Path:
    if yolo_version == YoloVersion.yolo_v7:
        return Path(WEIGHTS) / yolo7_model_gdrive_file

    return Path(WEIGHTS) / yolo8_model_gdrive_file


def get_link(yolo_version: YoloVersion) -> str:
    if yolo_version == YoloVersion.yolo_v7:
        return 'https://drive.google.com/uc?id=1U6zt4rOy2v3VrLjMrsqdHF_Y6k9INbib'

    return 'https://drive.google.com/uc?id=1pyuTy4w1GPaPZKwJP9aKI0PlqTVW5xw9'


def get_model_file(yolo_version: YoloVersion):
    output_path = get_local_path(yolo_version)

    output = str(output_path)

    if output_path.exists():
        print(f"{output} local exist")
        return output

    url = get_link(yolo_version)

    print(f"download {output} from {url}")

    gdown.download(url, output, quiet=False)

    return output


def download_test_video():
    output = str(ROOT / "testinfo")

    url = 'https://drive.google.com/uc?id=1YK0a3peuwdbvoZUAKciCvYM5KjKeizA6'

    print(f"download {output} from {url}")

    folders = gdown.download_folder(id="1YK0a3peuwdbvoZUAKciCvYM5KjKeizA6", output=output, quiet=False)
    # folders = gdown.download_folder(url=url, quiet=False)
    print(folders)


def post_process(test_func, track, num, w, h, bound_line, source) -> Result:
    # count humans

    humans_result = Result(0, 0, 0, [])

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

        except Exception as e:
            print_exception(e, "post processing")

    humans_result.file = str(Path(source).name)

    return humans_result


def run_single_video_yolo(source, yolo_info="7", conf=0.3, iou=0.45, test_func="timur",
                          tracker_type="fastdeepsort", log: bool = True) -> dict:
    print(f"yolo version = {yolo_info}")
    yolo_version = parse_yolo_version(yolo_info)

    if yolo_version is None:
        raise Exception(f"unsupported yolo version {yolo_info}")
    model = get_model_file(yolo_version)

    reid_weights = str(Path(WEIGHTS) / "osnet_x0_25_msmt17.pt")

    model = create_yolo_model(yolo_version, model)

    # tracker_type = "fastdeepsort"
    # tracker_type = "ocsort"

    # all_trackers = get_all_trackers_full_path()
    all_trackers = get_all_optune_trackers()
    tracker_config = all_trackers.get(tracker_type)

    if log:
        print(f"tracker_type = {tracker_type}")

    track = model.track(
        source=source,
        conf_threshold=conf,
        iou=iou,
        tracker_type=tracker_type,
        tracker_config=tracker_config,
        reid_weights=reid_weights,
        log=log
    )

    num, w, h, fps = get_camera(source)
    cameras_info = load_default_bound_line()
    bound_line = cameras_info.get(num)

    humans_result = post_process(test_func, track, num, w, h, bound_line, source)

    test_result_file = TEST_TRACKS_PATH

    test_results = TestResults(test_result_file)

    test_results.add_test(humans_result)

    test_res = test_results.compare_to_file_v2(output_folder=None)

    res_dic = \
        {
            "result": json.loads(results_to_json(humans_result)),
            "test_result": test_res,
            "num": num,
            "width": w,
            "height": h,
            "fps": fps,
            "bound_line": bound_line,
            "file": str(source)
        }

    return res_dic


if __name__ == '__main__':
    # download_test_video()

    # get_model_file(YoloVersion.yolo_v7)

    video_source = "d:\\AI\\2023\\corridors\\dataset-v1.1\\test\\"

    video_file = str(Path(video_source) / "1.mp4")

    # result = run_single_video_yolo(video_file, yolo_info="8ul", log=False)

    # print(result)

    res = Result(1, 2, 4, [])

    print(res.__dict__)

    str1 = results_to_json(res)
    print(str1)

    str2 = json.dumps(res.__dict__, indent=4)

    print(str2)

