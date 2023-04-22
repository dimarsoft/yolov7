import json
import os
from datetime import datetime
from pathlib import Path

from tools.labeltools import TrackWorker
from tools.save_txt_tools import yolo7_save_tracks_to_txt
from utils.general import set_logging
from utils.torch_utils import time_synchronized
from yolov7 import YOLO7


def detect_single_video_yolo7(weights, source, output_folder, classes=None, conf=0.1, save_txt=True, save_vid=False):
    print(f"start detect_single_video_yolo7 {source}")

    source_path = Path(source)
    text_path = Path(output_folder) / f"{source_path.stem}.txt"

    model = YOLO7(weights)

    detections = model.detect(
       source=source,
       conf_threshold=conf,
       classes=classes
    )

    if save_txt:
        print(f"save detections to: {text_path}")

        yolo7_save_tracks_to_txt(results=detections, txt_path=text_path, conf=conf)

    if save_vid:
        track_worker = TrackWorker(detections)
        t1 = time_synchronized()
        track_worker.create_video(source, output_folder, draw_class=True)
        t2 = time_synchronized()

        print(f"Processed '{source}' to {output_folder}: ({(1E3 * (t2 - t1)):.1f} ms)")


def run_detect_yolo7(model: str, source: str, output_folder,
                     files=None, classes=None, conf=0.3, save_txt=True, save_vid=False):
    """

    Args:
        save_txt: сохранять бб в файл
        files: если указана папка, но можно указать имена фай1лов,
                которые будут обрабатываться. ['1', '2' ...]
        classes: список классов, None все, [0, 1, 2....]
        conf: conf
        save_vid: Создаем видео c bb
        output_folder: путь к папке для результатов работы, txt
        source: путь к видео, если папка, то для каждого видео файла запустит
        model (str): модель для YOLO8
    """

    set_logging()

    source_path = Path(source)

    # в выходной папке создаем папку с сессией: дата_трекер туда уже сохраняем все файлы

    now = datetime.now()

    session_folder_name = f"{now.year:04d}_{now.month:02d}_{now.day:02d}_{now.hour:02d}_{now.minute:02d}_" \
                          f"{now.second:02d}_y7_detect"

    session_folder = str(Path(output_folder) / session_folder_name)

    try:
        os.makedirs(session_folder, exist_ok=True)
        print(f"Directory '{session_folder}' created successfully")
    except OSError as error:
        print(f"Directory '{session_folder}' can not be created. {error}")

    session_info = dict()

    session_info['model'] = str(Path(model).name)
    session_info['conf'] = conf
    session_info['save_vid'] = save_vid
    session_info['files'] = files
    session_info['classes'] = classes
    session_info['save_txt'] = save_txt

    session_info_path = str(Path(session_folder) / 'session_info.json')

    with open(session_info_path, "w") as session_info_file:
        json.dump(session_info, fp=session_info_file, indent=4)

    # test_results = TestResults(test_result_file)

    if source_path.is_dir():
        print(f"process folder: {source_path}")

        for entry in source_path.iterdir():
            # check if it is a file
            if entry.is_file() and entry.suffix == ".mp4":
                if files is None:
                    detect_single_video_yolo7(model, str(entry), session_folder,
                                              classes, conf, save_txt, save_vid)
                else:
                    if entry.stem in files:
                        detect_single_video_yolo7(model, str(entry), session_folder,
                                                  classes, conf, save_txt, save_vid)
    else:
        print(f"process file: {source_path}")
        detect_single_video_yolo7(model, str(source_path), session_folder,
                                  classes, conf, save_txt, save_vid)
