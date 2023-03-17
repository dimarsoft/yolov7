from ultralytics import YOLO
from pathlib import Path

from save_txt_tools import yolo8_save_tracks_to_txt


def create_video_with_track(results, source_video, output_file):
    import cv2

    input_video = cv2.VideoCapture(source_video)

    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    # ширина
    w = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    # высота
    h = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # количесто кадров в видео
    frames_in_video = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"input = {source_video}, w = {w}, h = {h}, fps = {fps}, frames_in_video = {frames_in_video}")

    output_video = cv2.VideoWriter(str(output_file), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    # считываем все фреймы из видео
    for i in range(frames_in_video):
        ret, frame = input_video.read()

        output_video.write(frame)

    output_video.release()
    input_video.release()


def run_single_video_yolo8(model, source, tracker, output_folder, conf=0.3, save_vid=False, save_vid2=False):
    print(f"start {source}")
    model = YOLO(model)

    track = model.track(
        source=source,
        stream=False,
        save=save_vid,
        conf=conf,
        tracker=tracker
    )
    source_path = Path(source)
    text_path = Path(output_folder) / f"{source_path.stem}.txt"

    print(f"save to: {text_path}")

    yolo8_save_tracks_to_txt(results=track, txt_path=text_path, conf=conf)

    if save_vid2:
        out_video_path = Path(output_folder) / f"{source_path.stem}.mp4"
        create_video_with_track(results=track, source_video=source, output_file=out_video_path)


def run_yolo8(model, source, tracker, output_folder,  conf=0.3, save_vid=False, save_vid2=False):
    """

    Args:
        conf: conf для трекера
        save_vid2: Создаем наше видео с центром человека
        save_vid (Bool): save для model.track. Yolo создает свое видео
        output_folder: путь к папке для результатов работы, txt
        tracker: трекер (botsort.yaml, bytetrack.yaml) или путь к своему файлу с настройками
        source: путь к видео, если папка, то для каждого видеофайла запустит
        model (object): модель для YOLO8
    """
    source_path = Path(source)

    if source_path.is_dir():
        print(f"process folder: {source_path}")

        for entry in source_path.iterdir():
            # check if it is a file
            if entry.is_file() and entry.suffix == ".mp4":
                run_single_video_yolo8(model, str(entry), tracker, output_folder, conf, save_vid, save_vid2)
    else:
        run_single_video_yolo8(model, source, tracker, output_folder, conf, save_vid, save_vid2)
