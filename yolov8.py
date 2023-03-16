from ultralytics import YOLO
from pathlib import Path
from save_txt_tools import yolo8_save_tracks_to_txt


def run_yolo8(model, source, tracker, output_folder):
    """

    Args:
        output_folder: путь к папке для результатов работы, txt
        tracker: трекер (botsort.yaml, bytetrack.yaml) или путь к своему файлу с настройками
        source: путь к видео
        model (object): модель для YOLO8
    """
    print(f"start {source}")
    model = YOLO(model)

    track = model.track(
        source=source,
        stream=False,
        save=False,
        conf=0.0,
        tracker=tracker
    )
    source_path = Path(source)
    text_path = Path(output_folder) / f"{source_path.stem}.txt"

    print(f"save to: {text_path}")

    yolo8_save_tracks_to_txt(results=track, txt_path=text_path, conf=0.0)
