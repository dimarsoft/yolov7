"""
Сохраняет результаты детекции YOLO8 и трекинга в текстовый файл

Формат: frame_index track_id class bbox_left bbox_top bbox_w bbox_h conf

bbox - в относительный величинах
"""
import json
from pathlib import Path

from exception_tools import print_exception
from track_objects import Track


def convert_txt_toy7(results, save_none_id=False):
    results_y7 = []

    for track in results:
        frame_index = track[0]
        xywhn = track

        bbox_w = xywhn[5]
        bbox_h = xywhn[6]
        bbox_left = xywhn[3]
        bbox_top = xywhn[4]

        track_id = int(track[1])
        cls = int(track[2])

        results_y7.append([frame_index, bbox_left, bbox_top, bbox_w, bbox_h, track_id, cls])

    return results_y7


def convert_toy7(results, save_none_id=False):
    results_y7 = []

    for frame_index, track in enumerate(results):
        track = track.cpu()
        if track.boxes is not None:
            for box in track.boxes:
                if save_none_id:
                    track_id = -1 if box.id is None else int(box.id)

                else:
                    if box.id is None:
                        continue
                    track_id = int(box.id)
                # if box.conf < conf:
                #    continue
                # print(frame_index, " ", box.id, ", cls = ", box.cls, ", ", box.xywhn)
                xywhn = box.xywhn.numpy()[0]
                # print(frame_index, " ", xywhn)
                bbox_w = xywhn[2]
                bbox_h = xywhn[3]
                bbox_left = xywhn[0] - bbox_w / 2
                bbox_top = xywhn[1] - bbox_h / 2
                bbox_r = xywhn[0] + bbox_w / 2
                bbox_b = xywhn[1] + bbox_h / 2
                # track_id = int(box.id)
                cls = int(box.cls)
                results_y7.append([frame_index, bbox_left, bbox_top, bbox_w, bbox_h, track_id, cls, box.conf])
    return results_y7


"""
 info = [frame_id,
                                float(detection[0]) / w, float(detection[1]) / h,
                                float(detection[2]) / w, float(detection[3]) / h,
                                int(detection[4]), int(detection[5]), float(detection[6])]
"""


def yolo8_save_tracks_to_txt(results, txt_path, conf=0.0, save_id=False):
    """

    Args:
        save_id:
        conf: элементы с conf менее указанной не сохраняются
        txt_path: текcтовый файл для сохранения
        results: результат работы модели
    """
    with open(txt_path, 'a') as text_file:
        for frame_index, track in enumerate(results):
            if track.boxes is not None:
                for box in track.boxes:
                    if box.id is None and not save_id:
                        continue
                    if box.conf < conf:
                        continue
                    # print(frame_index, " ", box.id, ", cls = ", box.cls, ", ", box.xywhn)
                    xywhn = box.xywhn.numpy()[0]
                    # print(frame_index, " ", xywhn)
                    bbox_w = xywhn[2]
                    bbox_h = xywhn[3]
                    bbox_left = xywhn[0] - bbox_w / 2
                    bbox_top = xywhn[1] - bbox_h / 2
                    track_id = int(box.id) if box.id is not None else -1
                    cls = int(box.cls)
                    text_file.write(('%g ' * 8 + '\n') % (frame_index, track_id, cls, bbox_left,
                                                          bbox_top, bbox_w, bbox_h, box.conf))


def yolo8_save_detection_to_txt(results, txt_path, conf=0.0, save_id=False):
    """

    Args:
        save_id:
        conf: элементы с conf менее указанной не сохраняются
        txt_path: текcтовый файл для сохранения
        results: результат работы модели
    """
    with open(txt_path, 'a') as text_file:
        for frame_index, track in enumerate(results):
            if track.boxes is not None:
                for box in track.boxes:
                    if box.id is None and not save_id:
                        continue
                    if box.conf < conf:
                        continue
                    # print(frame_index, " ", box.id, ", cls = ", box.cls, ", ", box.xywhn)
                    xywhn = box.xywhn.numpy()[0]
                    # print(frame_index, " ", xywhn)
                    bbox_w = xywhn[2]
                    bbox_h = xywhn[3]
                    bbox_left = xywhn[0] - bbox_w / 2
                    bbox_top = xywhn[1] - bbox_h / 2
                    track_id = int(box.id) if box.id is not None else -1
                    cls = int(box.cls)
                    text_file.write(('%g ' * 8 + '\n') % (frame_index, track_id, cls, bbox_left,
                                                          bbox_top, bbox_w, bbox_h, box.conf))


def yolo7_save_tracks_to_txt(results, txt_path, conf=0.0):
    """

    Args:
        conf: элементы с conf менее указанной не сохраняются
        txt_path: текстовый файл для сохранения
        results: результат работы модели
    """
    with open(txt_path, 'a') as text_file:
        for track in results:
            if track[7] < conf:
                continue
            # print(frame_index, " ", box.id, ", cls = ", box.cls, ", ", box.xywhn)
            xywhn = track[1:5]
            # print(frame_index, " ", xywhn)
            bbox_w = xywhn[2]
            bbox_h = xywhn[3]
            bbox_left = xywhn[0]
            bbox_top = xywhn[1]
            track_id = int(track[5])
            cls = int(track[6])
            text_file.write(('%g ' * 8 + '\n') % (track[0], track_id, cls, bbox_left,
                                                  bbox_top, bbox_w, bbox_h, track[7]))


def yolo7_save_tracks_to_json(results, json_file, conf=0.0):
    """

    Args:
        conf: элементы с conf менее указанной не сохраняются
        json_file: json файл для сохранения
        results: результат работы модели
    """

    results_json = []

    for track in results:
        object_conf = track[7]
        if object_conf < conf:
            continue
        ltwhn = track[1:5]
        track_id = int(track[5])
        cls = int(track[6])
        frame_index = track[0]

        track = Track(ltwhn, cls, object_conf, frame_index, track_id)

        results_json.append(track)

    with open(json_file, "w") as write_file:
        write_file.write(json.dumps(results_json, indent=4, sort_keys=True, default=lambda o: o.__dict__))


def yolo_load_detections_from_txt(txt_path):
    import pandas as pd

    try:
        if Path(txt_path).stat().st_size > 0:
            df = pd.read_csv(txt_path, delimiter=" ", dtype=float, header=None)
        else:
            # если файл пустой, создаем пустой df, f nj pd.read_csv exception выдает
            df = pd.DataFrame(dtype=float, columns=[0, 1, 2, 3, 4, 5, 6, 7])
        # df = pd.DataFrame(df, columns=['frame', 'id', 'class', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf'])
    except Exception as ex:
        print_exception(ex, f"yolo_load_detections_from_txt: '{str(txt_path)}'")
        df = pd.DataFrame(dtype=float, columns=[0, 1, 2, 3, 4, 5, 6, 7])

    return df
