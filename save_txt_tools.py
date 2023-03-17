"""
Сохраняет реззультаты детекции YOLO8 и трекинга в текстовый файл

Формат: frame_index track_id class bbox_left bbox_top bbox_w bbox_h conf

bbox - в относительный величинах
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