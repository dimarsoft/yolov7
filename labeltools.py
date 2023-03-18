from enum import Enum
from pathlib import Path
from typing import List

import cv2

from resultools import Result


# класс
class Labels(Enum):
    human = 0
    helmet = 1
    uniform = 2


# положение человека отсительно турникета
class HumanPos(Enum):
    below = 0  # ниже
    above = 1  # выше
    near = 2  # в пределах турникета


# цвета обектов

label_colors = {
    Labels.human: (255, 255, 0),
    Labels.helmet: (255, 0, 255),
    Labels.uniform: (255, 255, 255)
}

# 10 различных цветов для объектов
human_colors = \
    [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (127, 0, 0),
        (0, 127, 0),
        (0, 0, 127),
        (64, 0, 0),
        (0, 64, 0),
        (0, 0, 64),
        (127, 127, 0)
    ]


def get_color(item_id: int):
    item_id = item_id % len(human_colors)
    return human_colors[item_id]


class Bbox:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = min(x1, x2)
        self.x2 = max(x1, x2)
        self.y1 = min(y1, y2)
        self.y2 = max(y1, y2)
        self.box = [self.x1, self.y1, self.x2, self.y2]
        self.width = abs(self.x1 - self.x2)
        self.height = abs(self.y1 - self.y2)

    @property
    def area(self):
        """
        Calculates the surface area. useful for IOU!
        """
        return (self.x2 - self.x1 + 1) * (self.y2 - self.y1 + 1)

    def intersection(self, bbox):
        x1 = max(self.x1, bbox.x1)
        y1 = max(self.y1, bbox.y1)
        x2 = min(self.x2, bbox.x2)
        y2 = min(self.y2, bbox.y2)
        intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
        return intersection

    def iou(self, bbox):
        intersection = self.intersection(bbox)

        iou = intersection / float(self.area + bbox.area - intersection)
        # return the intersection over union value
        return iou


class DetectedLabel:
    def __init__(self, label, x, y, width, height):
        self.label = label
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.human_pos = None

    def label_str(self):
        if self.label is Labels.human:
            return "human"
        if self.label is Labels.uniform:
            return "uniform"
        if self.label is Labels.helmet:
            return "helmet"
        return ""


class DetectedTrackLabel(DetectedLabel):
    def __init__(self, label, x, y, width, height, track_id, frame):
        super(DetectedTrackLabel, self).__init__(label, x, y, width, height)

        self.track_id = track_id
        self.frame = frame
        self.track_color = None

    def get_caption(self):
        return f"{self.track_id}: {self.label_str()}"


def draw_track_on_frame(frame, draw_rect, frame_w, frame_h, frame_info: DetectedTrackLabel):
    # if frame_info.labels is not None:
    lab = frame_info

    # cv2.putText(frame, f"{lab.frame} ", (0, 40), 0, 1, (255, 255, 255), 2, cv2.LINE_AA)

    hh = int(lab.height * frame_h)
    ww = int(lab.width * frame_w)

    x = int(lab.x * frame_w - ww / 2)
    y = int(lab.y * frame_h - hh / 2)

    font_scale = 0.4

    # рамка объекта

    if draw_rect:
        if lab.label is Labels.human:
            if lab.track_color is not None:
                caption_color = lab.track_color
            else:
                caption_color = (255, 255, 255)
            cv2.putText(frame, frame_info.get_caption(), (x, y), 0, font_scale, caption_color, 1, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + ww, y + hh), label_colors[lab.label], 1)

    # если человек, то рисуем центр масс
    if lab.label is Labels.human:
        x = int(x + ww / 2)
        y = int(y + hh / 2)

        if lab.human_pos is HumanPos.above:
            color = (0, 0, 255)
        else:
            if lab.human_pos is HumanPos.below:
                color = (0, 255, 0)
            else:
                color = (255, 255, 0)

        if lab.track_color is not None:
            cv2.circle(frame, (x, y), 10, lab.track_color, -1)
            cv2.circle(frame, (x, y), 5, color, -1)
            # cv2.putText(frame, f"{lab.track_id}", (x, y), 0, 1, lab.track_color, 1, cv2.LINE_AA)
        else:
            cv2.circle(frame, (x, y), 5, color, -1)
            # cv2.putText(frame, f"{lab.track_id}", (x, y), 0, 1, color, 1, cv2.LINE_AA)


class TrackWorker:
    def __init__(self, track_result, a=-0.2, b=0.68) -> None:
        # турникет, потом вынести в настройку
        self.line_a = a
        self.line_b = b

        self.track_result = track_result
        self.track_labels = TrackWorker.convert_tracks_to_list(track_result)
        self.test_track_human(self.track_labels)
        self.fill_track_color()

    # получить координату y для х
    def get_y(self, x):
        return self.line_a * x + self.line_b

    def test_track_human(self, track_list):
        for track in track_list:
            if track.label is Labels.human:
                self.test_human(track)

    # проверка человека: выше/ниже турникета
    def test_human(self, label):
        y_turniket = self.get_y(label.x)

        if abs(y_turniket - label.y) <= 0.008:
            label.human_pos = HumanPos.near
        else:
            if y_turniket > label.y:
                label.human_pos = HumanPos.above
            else:
                label.human_pos = HumanPos.below

    # заделить трек по id
    # получаем словать: ключ = id, значение список треков по это id
    # также указываем класс
    @staticmethod
    def get_tracks_info_by_id(tracks, label_type: Labels = Labels.human):
        tracks_by_id = dict()

        for track in tracks:
            if track.label is label_type:
                by_id = tracks_by_id.get(track.track_id)
                if by_id is None:
                    by_id = []
                    tracks_by_id[track.track_id] = by_id
                by_id.append(track)

        return tracks_by_id

    def fill_track_color(self):
        tracks_info_by_id = self.get_tracks_info_by_id(self.track_labels, label_type=Labels.human)

        # наложение трека
        color_id = 0

        for key in tracks_info_by_id:
            tracks = tracks_info_by_id[key]

            object_color = get_color(color_id)

            for track in tracks:
                track.track_color = object_color

            color_id += 1

    # # Process detections [f, x1, y1, x2, y2, track_id, class_id, conf]
    @staticmethod
    def convert_tracks_to_list(results: List):
        track_list = []
        for track in results:
            frame_index = track[0]
            xywhn = track[1:5]

            bbox_w = abs(xywhn[2] - xywhn[0])
            bbox_h = abs(xywhn[3] - xywhn[1])
            bbox_left = min(xywhn[0], xywhn[2])
            bbox_top = min(xywhn[1], xywhn[3])

            track_id = int(track[5])
            cls = int(track[6])
            x_center = bbox_left + bbox_w / 2
            y_center = bbox_top + bbox_h / 2

            track_list.append(
                DetectedTrackLabel(Labels(cls), x_center, y_center, bbox_w, bbox_h, track_id, frame_index))

        return track_list

    def draw_turnic_on_frame(self, frame, frame_w, frame_h):
        # турникет рисуем один раз
        y1 = int(self.get_y(0) * frame_h)
        y2 = int(self.get_y(1) * frame_h)
        cv2.line(frame, (0, y1), (frame_w, y2), (0, 0, 255), 1)

    def create_video(self, source_video, output_folder):
        input_video = cv2.VideoCapture(str(source_video))

        fps = int(input_video.get(cv2.CAP_PROP_FPS))
        # ширина
        w = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        # высота
        h = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # количесто кадров в видео
        frames_in_video = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

        results = []

        for frame_id in range(frames_in_video):
            ret, frame = input_video.read()
            results.append(frame)
        input_video.release()

        for i in range(frames_in_video):
            self.draw_turnic_on_frame(results[i], w, h)

        for label in self.track_labels:
            draw_track_on_frame(results[label.frame], True, w, h, label)

        output_video_path = str(Path(output_folder) / Path(source_video).name)
        output_video = cv2.VideoWriter(
            output_video_path, cv2.VideoWriter_fourcc(*'mp4v'),
            fps, (w, h))
        # запись в выходной файл
        for i in range(frames_in_video):
            output_video.write(results[i])

        output_video.release()

    def test_humans(self):
        return Result(0, 0, 0, [])
