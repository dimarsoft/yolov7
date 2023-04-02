from enum import Enum
from pathlib import Path
from typing import Dict

from post_processing.timur import load_bound_line

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
CONFIG_ROOT = ROOT / 'cfg'
TEST_ROOT = ROOT / 'testinfo'
# Файл с настройками камер для каждого видео.
# Пока информация о турникете
CAMERAS_PATH = CONFIG_ROOT / 'camera_config.json'
TEST_TRACKS_PATH = TEST_ROOT / 'all_track_results.json'
WEIGHTS = ROOT / 'weights'


class YoloVersion(Enum):
    yolo_v7 = 7,
    yolo_v8 = 8,    # код из yolo8_tracking
    yolo_v8ul = 81  # из пакета ultralytics


def parse_yolo_version(yolo_version):
    if isinstance(yolo_version, YoloVersion):
        return yolo_version

    if isinstance(yolo_version, int):
        if yolo_version == 7:
            return YoloVersion.yolo_v7
        if yolo_version == 8:
            return YoloVersion.yolo_v8
    if isinstance(yolo_version, str):
        if yolo_version == '7' or yolo_version == 'y7' or yolo_version == 'yolov7' or yolo_version == 'yolo7':
            return YoloVersion.yolo_v7
        if yolo_version == '8' or yolo_version == 'y8' or yolo_version == 'yolov8' or yolo_version == 'yolo8':
            return YoloVersion.yolo_v8
        if yolo_version == '8ul' or yolo_version == 'y8ul' or yolo_version == 'yolov8ul' or yolo_version == 'yolo8ul':
            return YoloVersion.yolo_v8ul

    return None


def load_default_bound_line():
    """
    Загрузка информации по камерам по видео файлам.


    Returns:
        Словарь настроек
        Ключи(str) имя файла
        Значение = координаты турникета

    """
    print(f"read camera info from '{CAMERAS_PATH}'")

    return load_bound_line(CAMERAS_PATH)


def get_all_trackers():
    all_trackers = \
        {
            'sort': 'trackers/sort/configs/sort.yaml',
            'botsort': 'trackers/botsort/configs/botsort.yaml',
            'bytetrack': 'trackers/bytetrack/configs/bytetrack.yaml',
            'deepsort': 'trackers/deep_sort/configs/deepsort.yaml',

            'ocsort': 'trackers/ocsort/configs/ocsort.yaml',
            'strongsort': 'trackers/strongsort/configs/strongsort.yaml',
            'fastdeepsort': 'trackers/fast_deep_sort/configs/fastdeepsort.yaml',
            'norfair': 'trackers/NorFairTracker/configs/norfair_track.yaml',
        }

    return all_trackers


def get_all_trackers_full_path():
    all_trackers = get_all_trackers()

    for key in all_trackers.keys():
        path = ROOT / all_trackers[key]

        all_trackers[key] = str(path)

    return all_trackers


def get_select_trackers(trackers_names, trackers) -> dict:
    """

    Args:
        trackers (Dict): Словарь трекеров из которых выбираем
        trackers_names (List): Список строк с именами трекеров

    Returns: Словарь выбранных трекеров
    """
    selected = {}
    for key in trackers_names:
        selected[key] = trackers[key]

    return selected
