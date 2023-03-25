from pathlib import Path

from post_processing.timur import load_bound_line

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
CONFIG_ROOT = ROOT / 'cfg'
TEST_ROOT = ROOT / 'testinfo'
# Файл с настройками камер для каждого видео.
# Пока информация о турникете
CAMERAS_PATH = CONFIG_ROOT / 'camera_config.json'
TEST_TRACKS_PATH = TEST_ROOT / 'all_track_results.json'


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

