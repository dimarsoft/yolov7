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
