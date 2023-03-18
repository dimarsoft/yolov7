from pathlib import Path

from labeltools import TrackWorker
from save_txt_tools import yolo7_save_tracks_to_txt
from yolov7 import YOLO7


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


def run_single_video_yolo7(model, source, tracker_type: str, tracker_config, output_folder,
                           reid_weights, conf=0.3, save_vid=False):
    print(f"start {source}")

    source_path = Path(source)
    text_path = Path(output_folder) / f"{source_path.stem}.txt"

    model = YOLO7(model)

    track = model.track(
        source=source,
        conf=conf,
        tracker_type=tracker_type,
        tracker_config=tracker_config,
        reid_weights=reid_weights
    )

    print(f"save to: {text_path}")

    yolo7_save_tracks_to_txt(results=track, txt_path=text_path, conf=conf)

    if save_vid:
        track_worker = TrackWorker(track)
        track_worker.draw_track_on_frame(source, output_folder)


def run_yolo7(model, source, tracker_type: str, tracker_config, output_folder, reid_weights, conf=0.3, save_vid=False):
    """

    Args:
        reid_weights:
        conf: conf для трекера
        save_vid: Создаем наше видео с центром человека
        output_folder: путь к папке для результатов работы, txt
        tracker_type: трекер (botsort.yaml, bytetrack.yaml)
        tracker_config: путь к своему файлу с настройками
        source: путь к видео, если папка, то для каждого видеофайла запустит
        model (object): модель для YOLO7
    """
    source_path = Path(source)

    if source_path.is_dir():
        print(f"process folder: {source_path}")

        for entry in source_path.iterdir():
            # check if it is a file
            if entry.is_file() and entry.suffix == ".mp4":
                run_single_video_yolo7(model, str(entry), tracker_type, tracker_config, output_folder,
                                       reid_weights, conf, save_vid)
    else:
        run_single_video_yolo7(model, source, tracker_type, tracker_config, output_folder,
                               reid_weights, conf, save_vid)


def run_example():
    model = "D:\\AI\\2023\\models\\Yolov7\\25.02.2023_dataset_1.1_yolov7_best.pt"
    video_source = "d:\\AI\\2023\\corridors\\dataset-v1.1\\test\\1.mp4"

    tracker_config = "./trackers/strongsort/configs/strongsort.yaml"
    output_folder = "d:\\AI\\2023\\corridors\\dataset-v1.1\\"

    # run_yolo7(model, video_source, "strongsort", tracker_config, output_folder)

    tracker_config = "trackers/deep_sort/configs/deepsort.yaml"
    reid_weights = "mars-small128.pb"
    run_yolo7(model, video_source, "deepsort", tracker_config, output_folder, reid_weights)

    tracker_config = "trackers/botsort/configs/botsort.yaml"
    reid_weights = "osnet_x0_25_msmt17.pt"
    # run_yolo7(model, video_source, "botsort", tracker_config, output_folder, reid_weights)

    tracker_config = "trackers/ocsort/configs/ocsort.yaml"
    reid_weights = "osnet_x0_25_msmt17.pt"
    # run_yolo7(model, video_source, "ocsort", tracker_config, output_folder, reid_weights)

    tracker_config = "trackers/bytetrack/configs/bytetrack.yaml"
    reid_weights = "osnet_x0_25_msmt17.pt"
    # run_yolo7(model, video_source, "bytetrack", tracker_config, output_folder, reid_weights)


if __name__ == '__main__':
    run_example()
