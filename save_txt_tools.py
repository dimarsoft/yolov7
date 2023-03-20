"""
Сохраняет реззультаты детекции YOLO8 и трекинга в текстовый файл

Формат: frame_index track_id class bbox_left bbox_top bbox_w bbox_h conf

bbox - в относительный величинах
"""


def convert_toy7(results):
    results_y7 = []

    for frame_index, track in enumerate(results):
        if track.boxes is not None:
            for box in track.boxes:
                if box.id is None:
                    continue
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
                track_id = int(box.id)
                cls = int(box.cls)
                results_y7.append([frame_index, bbox_left, bbox_top, bbox_w, bbox_h, track_id, cls, box.conf])
    return results_y7


"""
 info = [frame_id,
                                float(detection[0]) / w, float(detection[1]) / h,
                                float(detection[2]) / w, float(detection[3]) / h,
                                int(detection[4]), int(detection[5]), float(detection[6])]
"""


def yolo8_save_tracks_to_txt(results, txt_path, conf=0.0):
    """

    Args:
        conf: элементы с conf менее указанной не сохраняются
        txt_path: тектосвый файл для сохрения
        results: результат работы модели
    """
    with open(txt_path, 'a') as text_file:
        for frame_index, track in enumerate(results):
            if track.boxes is not None:
                for box in track.boxes:
                    if box.id is None:
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
                    track_id = int(box.id)
                    cls = int(box.cls)
                    text_file.write(('%g ' * 8 + '\n') % (frame_index, track_id, cls, bbox_left,
                                                          bbox_top, bbox_w, bbox_h, box.conf))


def yolo7_save_tracks_to_txt(results, txt_path, conf=0.0):
    """

    Args:
        conf: элементы с conf менее указанной не сохраняются
        txt_path: тектосвый файл для сохрения
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
