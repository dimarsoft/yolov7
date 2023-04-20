from pathlib import Path
from typing import Union, Optional

import cv2

from configs import load_bound_line, CAMERAS_PATH, get_bound_line
from labeltools import draw_label_text
from path_tools import get_video_files
from post_processing.timur import get_camera


def save_frames(path_to_video: Union[str, Path], output_folder: Union[str, Path], max_frames: int = -1):
    input_video = cv2.VideoCapture(str(path_to_video))

    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    # ширина
    w = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    # высота
    h = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # количество кадров в видео
    frames_in_video = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

    path_to_video = Path(path_to_video)

    if max_frames > 0:
        max_frames = min(max_frames, frames_in_video)
    else:
        max_frames = frames_in_video

    output_folder = Path(output_folder)

    for frame_id in range(max_frames):
        ret, frame = input_video.read()
        image_path = output_folder / f"{path_to_video.stem}_frame_{frame_id}_{w}_{h}.jpg"
        cv2.imwrite(str(image_path), frame)

    input_video.release()


def draw_turnic_on_frame(frame, bound_line):
    p1 = bound_line[0]
    p2 = bound_line[1]
    y1 = int(p1[1])
    y2 = int(p2[1])
    cv2.line(frame, (p1[0], y1), (p2[0], y2), (0, 255, 0), 2)


def draw_frame_info(frame, caption: str, lw: int) -> None:
    draw_label_text(frame, (0, 0), caption, lw, (0, 0, 0))


def create_video(source_video: Union[str, Path], output_folder: Union[str, Path], bound_line,
                 draw_frame_index: bool = False) -> None:
    """
    Создание нового видео с отображением турникета и номера кадра
    Args:
        source_video:  Путь к видео файлу
        output_folder: Папка куда сохранять новое видео
        bound_line: Линия турнике
        draw_frame_index(bool): Флаг отображения номера кадра (нумерация с нуля)

    Returns:

    """
    input_video = cv2.VideoCapture(str(source_video))

    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    # ширина
    w = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    # высота
    h = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # количество кадров в видео
    frames_in_video = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

    output_video_path = str(Path(output_folder) / Path(source_video).name)
    output_video = cv2.VideoWriter(
        output_video_path, cv2.VideoWriter_fourcc(*'mp4v'),
        fps, (w, h))

    lw = int(round(3 * h / 640.0))

    for frame_id in range(frames_in_video):
        ret, frame = input_video.read()
        frame_info = f"{frame_id}/{frames_in_video}"
        if ret:
            draw_turnic_on_frame(frame, bound_line)
            if draw_frame_index:
                draw_frame_info(frame, frame_info, lw=lw)
            output_video.write(frame)

        print(frame_info)

    input_video.release()
    output_video.release()


def create_video_with_turn(source: Union[str, Path], out_folder: Union[str, Path]):
    bound_lines = load_bound_line(CAMERAS_PATH)
    camera_num, w, h, fps = get_camera(source)
    bound_line = get_bound_line(bound_lines, camera_num)

    create_video(source, out_folder, bound_line, draw_frame_index=True)


def create_videos_with_turn_in_folder(
        source: Union[str, Path],
        out_folder: Union[str, Path],
        files: Optional[list[str]] = None):
    """
    Функция создания файлов с отображением турникета и номеров кадров
    Args:
        source: Путь к папке с видео файлами
        out_folder: Папка для сохранения новых файлов
        files: Список номеров файлов, если None то все

    Returns:

    """
    bound_lines = load_bound_line(CAMERAS_PATH)

    # список файлов с видео для обработки
    list_of_videos = get_video_files(source, files)

    total_videos = len(list_of_videos)

    for i, item in enumerate(list_of_videos):
        print(f"process file: {i + 1}/{total_videos} {item}")

        camera_num, w, h, fps = get_camera(item)
        bound_line = get_bound_line(bound_lines, camera_num)

        create_video(item, out_folder, bound_line, draw_frame_index=True)


if __name__ == '__main__':
    # save_frames("D:\\48.mp4", "d:\\AI\\2023\\18.04.2023\\", 10)
    # draw_turn("D:\\48.mp4", "d:\\AI\\2023\\18.04.2023\\")
    # create_video_with_turn("D:\\44.mp4", "d:\\AI\\2023\\18.04.2023\\")
    # create_video_with_turn("D:\\10.mp4", "d:\\AI\\2023\\18.04.2023\\")
    # create_videos_with_turn_in_folder("D:\\", "d:\\AI\\2023\\18.04.2023\\")
    pass
