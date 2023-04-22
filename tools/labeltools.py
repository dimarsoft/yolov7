from enum import Enum
from enum import IntEnum

from pathlib import Path
from typing import List, Optional

import cv2
from ultralytics.yolo.utils.plotting import Colors

from tools.count_results import Result, Deviation


# класс
class Labels(IntEnum):
    # Человек
    human: int = 0
    # Каска
    helmet: int = 1
    # Жилет
    uniform: int = 2


# положение человека относительно турникета
class HumanPos(Enum):
    below = 0  # ниже
    above = 1  # выше
    near = 2  # в пределах турникета


# цвета объектов

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

# цвета возьмем стандартные
object_colors = Colors()


def get_color(item_id: int):
    return object_colors(item_id, True)
    # item_id = item_id % len(human_colors)
    # return human_colors[item_id]


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
    def __init__(self, label, x, y, width, height, conf):
        """
        Объект обнаруженный детекцией
        Args:
            label: класс
            x:
            y:
            width:
            height:
            conf: вероятность
        """
        self.label = label
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.human_pos = None
        self.conf = float(conf)

    def label_str(self) -> str:
        if self.label is Labels.human:
            return f"human: {self.conf:.2f}"
        if self.label is Labels.uniform:
            return "uniform"
        if self.label is Labels.helmet:
            return "helmet"
        return ""


class DetectedTrackLabel(DetectedLabel):
    def __init__(self, label, x, y, width, height, track_id, frame, conf):
        """
        Объект обнаруженный детекцией и прошедший через трекер
        Args:
            label:
            x:
            y:
            width:
            height:
            track_id: ИД трека, -1 если его нет
            frame:
            conf:
        """
        super(DetectedTrackLabel, self).__init__(label, x, y, width, height, conf)

        self.track_id = track_id
        self.frame = frame
        self.track_color = None

    def get_caption(self):
        if self.track_id < 0:
            return self.label_str()
        return f"{self.track_id}: {self.label_str()}"


def draw_label_text(im, p1, label, lw, color, txt_color=(255, 255, 255)):
    tf = max(lw - 1, 1)  # font thickness
    w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
    outside = p1[1] - h >= 3
    # p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3

    p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
    cv2.rectangle(im, p1, p2, color, -1, cv2.LINE_AA)  # filled

    cv2.putText(im,
                label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                0,
                lw / 3,
                txt_color,
                thickness=tf,
                lineType=cv2.LINE_AA)


def draw_track_on_frame(frame, draw_rect, frame_w,
                        frame_h,
                        frame_info: DetectedTrackLabel,
                        draw_center=False,
                        draw_class=False):
    # if frame_info.labels is not None:
    lab = frame_info

    # cv2.putText(frame, f"{lab.frame} ", (0, 40), 0, 1, (255, 255, 255), 2, cv2.LINE_AA)

    hh = int(lab.height * frame_h)
    ww = int(lab.width * frame_w)

    x = int(lab.x * frame_w - ww / 2)
    y = int(lab.y * frame_h - hh / 2)

    font_scale = 0.4

    # рамка объекта

    label_color = object_colors(lab.label, True)
    line_width = 2

    if draw_rect:
        if lab.label is Labels.human:

            if draw_class or frame_info.track_id < 0:
                caption = frame_info.get_caption()
            else:
                caption = f"{frame_info.track_id}"

            draw_label_text(frame, (x, y), caption, line_width, label_color)

        # cv2.rectangle(frame, (x, y), (x + ww, y + hh), label_colors[lab.label], 1)
        cv2.rectangle(frame, (x, y), (x + ww, y + hh), label_color, line_width)

    # если человек, то рисуем центр масс

    if draw_center and lab.label is Labels.human:
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


def find_frame(frame, tracks):
    for t in tracks:
        if t.frame == frame:
            return t
    return None


class NearItem:
    def __init__(self, start_pos, start_frame, end_frame, start_i, end_i):
        self.start_pos = start_pos
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.start_i = start_i
        self.end_i = end_i


def get_status(has_helmet: bool, has_uniform: bool) -> int:
    """
    Получить код нарушения
    Args:
        has_helmet: Есть каска?
        has_uniform: Есть жилет

    Returns:

    """
    status = 0
    if not has_helmet and not has_uniform:
        status = 1
    else:
        if has_helmet and not has_uniform:
            status = 2
        else:
            if not has_helmet and has_uniform:
                status = 3
    return status


class TrackWorker:
    def __init__(self, track_result, a=-0.2, b=0.68) -> None:
        # турникет, потом вынести в настройку
        self.line_a = a
        self.line_b = b

        self.track_result = track_result
        self.track_labels = TrackWorker.convert_tracks_to_list(track_result)
        self.test_track_human(self.track_labels)
        self.fill_track_color()

    @staticmethod
    def convert_to_bb(label: DetectedLabel):
        x1 = label.x - label.width / 2
        y1 = label.y - label.height / 2

        x2 = x1 + label.width
        y2 = y1 + label.height

        return Bbox(x1, y1, x2, y2)

    def check_bb(self, h_bb, t_bb):
        bb_main = self.convert_to_bb(h_bb)
        bb_test = self.convert_to_bb(t_bb)

        # print(f"bb_main = {bb_main}, bb_test = {bb_test}")

        if (bb_test.x1 < bb_main.x1) and (bb_test.x2 < bb_main.x1):
            return False

        if (bb_test.x1 > bb_main.x2) and (bb_test.x2 > bb_main.x2):
            return False

        if (bb_test.y1 < bb_main.y1) and (bb_test.y2 < bb_main.y1):
            return False

        if (bb_test.y1 > bb_main.y2) and (bb_test.y2 > bb_main.y2):
            return False

        iou = bb_test.iou(bb_main)

        # print(f"iou = {iou}")

        return True

    def find_near_in_track(self, tracks_human, track, near_info):

        for frame in range(near_info[0], near_info[1] + 1):
            # print(f"find_near_in_track: frame = {frame}")
            h_bb = find_frame(frame, tracks_human)
            t_bb = find_frame(frame, track)

            if h_bb is not None and t_bb is not None:
                # print(f"find_near_in_track: frame = {frame}, found")

                if self.check_bb(h_bb, t_bb):
                    return True

        return False

    def find_near(self, tracks_human, tracks_by_id, near_info):
        for track_id in tracks_by_id:
            tracks = tracks_by_id[track_id]
            # print(f"find_near = {track_id}")
            if self.find_near_in_track(tracks_human, tracks, near_info):
                return track_id
        return None

    @staticmethod
    def invert_human_pos(human_pos):
        if human_pos == HumanPos.above:
            return HumanPos.below
        if human_pos == HumanPos.below:
            return HumanPos.above
        return human_pos

    def get_near_v2(self, start, tracks) -> Optional[NearItem]:
        start_pos = tracks[start].human_pos
        end_pos = self.invert_human_pos(start_pos)

        end = len(tracks)

        start_frame = None
        start_i = None

        end_frame = None
        end_i = None

        for i in range(start, end):
            if tracks[i].human_pos is HumanPos.near:
                start_frame = tracks[i].frame
                start_i = i
                break
        if start_i is not None:
            for i in range(start_i + 1, end):
                if tracks[i].human_pos is end_pos:
                    end_frame = tracks[i].frame
                    end_i = i
                    break
            if end_i is not None:
                return NearItem(start_pos, start_frame, end_frame, start_i, end_i)
            else:
                return NearItem(start_pos, start_frame, tracks[end - 1].frame, start_i, end - 1)

        for i in range(start + 1, end):
            if tracks[i].human_pos is end_pos:
                start_frame = tracks[i].frame
                start_i = i
                break

        if start_i is not None:
            return NearItem(start_pos, start_frame, start_frame, start_i, start_i)

        return None

    def get_near_items(self, tracks) -> List:

        start = 0
        end = len(tracks)

        items = []

        while start < end:
            item = self.get_near_v2(start, tracks)
            if item is None:
                break

            items.append(item)

            start = item.end_i + 1

        return items

    # получить координату(frame) прохождения турникета
    def get_near(self, tracks):
        start_frame = None
        end_frame = None
        for track in tracks:
            if track.human_pos is HumanPos.near:
                start_frame = track.frame
                break

        reversed_tracks = tracks[::-1]
        for track in reversed_tracks:
            if track.human_pos is HumanPos.near:
                end_frame = track.frame
                break

        if start_frame is not None and end_frame is not None:
            return [start_frame, end_frame]

        start_frame = None
        end_frame = None
        pos = tracks[0].human_pos
        for track in tracks:
            if track.human_pos is not pos:
                start_frame = track.frame
                break

        for track in reversed_tracks:
            if track.human_pos is pos:
                end_frame = track.frame
                break
        if start_frame is not None and end_frame is not None:
            return [start_frame, end_frame]

        return None

    def get_humans_counter(self):
        track_list = self.track_labels
        # треки с людьми
        track_human_by_id = self.get_tracks_info_by_id(track_list, label_type=Labels.human)

        # print(f"track_human_by_id = {len(track_human_by_id)}")

        # треки с касками
        tracks_helmet_by_id = self.get_tracks_info_by_id(track_list, label_type=Labels.helmet)

        # print(f"tracks_helmet_by_id = {len(tracks_helmet_by_id)}")

        # треки с жилетами
        tracks_uniform_by_id = self.get_tracks_info_by_id(track_list, label_type=Labels.uniform)

        # print(f"tracks_uniform_by_id = {len(tracks_uniform_by_id)}")

        counter = 0
        counter_in = 0
        counter_out = 0
        violations = []
        for track_id in track_human_by_id:
            tracks = track_human_by_id[track_id]
            near_items = self.get_near_items(tracks)

            for near_item in near_items:

                if near_item.start_pos == HumanPos.above:
                    counter_in += 1
                else:
                    counter_out += 1

                near_info = [near_item.start_frame, near_item.end_frame]

                # if near_info is not None:
                counter += 1

                # print(f"human = {id}, {near_info}")

                has_helmet = False
                has_uniform = False

                helmet_found_id = self.find_near(tracks, tracks_helmet_by_id, near_info)

                if helmet_found_id is not None:
                    # нашли каску
                    has_helmet = True
                    # удаляем трек с каской
                    del tracks_helmet_by_id[helmet_found_id]

                uniform_found_id = self.find_near(tracks, tracks_uniform_by_id, near_info)

                if uniform_found_id is not None:
                    # нашли каску
                    has_uniform = True
                    # удаляем трек с каской
                    del tracks_uniform_by_id[uniform_found_id]

                status = get_status(has_helmet=has_helmet, has_uniform=has_uniform)

                if status > 0:
                    violations.append(Deviation(near_info[0], near_info[1], status))

        return Result(counter_in + counter_out, counter_in, counter_out, violations)

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

    # разделить трек по id
    # получаем словарь: ключ = id, значение список треков по это id
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

            bbox_w = xywhn[2]
            bbox_h = xywhn[3]
            bbox_left = xywhn[0]
            bbox_top = xywhn[1]

            track_id = int(track[5])
            cls = int(track[6])
            x_center = bbox_left + bbox_w / 2
            y_center = bbox_top + bbox_h / 2

            conf = track[7]

            track_list.append(
                DetectedTrackLabel(Labels(cls), x_center, y_center, bbox_w, bbox_h,
                                   track_id, frame_index, conf=conf))

        return track_list

    def draw_turnic_on_frame(self, frame, frame_w, frame_h):
        # турникет рисуем один раз
        y1 = int(self.get_y(0) * frame_h)
        y2 = int(self.get_y(1) * frame_h)
        cv2.line(frame, (0, y1), (frame_w, y2), (0, 0, 255), 1)

    def create_video(self, source_video, output_folder, draw_class=False):
        input_video = cv2.VideoCapture(str(source_video))

        fps = int(input_video.get(cv2.CAP_PROP_FPS))
        # ширина
        w = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        # высота
        h = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # количество кадров в видео
        frames_in_video = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

        results = []

        for frame_id in range(frames_in_video):
            ret, frame = input_video.read()
            results.append(frame)
            self.draw_turnic_on_frame(frame, w, h)
        input_video.release()

#        for i in range(frames_in_video):
#            self.draw_turnic_on_frame(results[i], w, h)

        for label in self.track_labels:
            draw_track_on_frame(results[int(label.frame)], True, w, h, label, draw_class=draw_class)

        output_video_path = str(Path(output_folder) / Path(source_video).name)
        output_video = cv2.VideoWriter(
            output_video_path, cv2.VideoWriter_fourcc(*'mp4v'),
            fps, (w, h))
        # запись в выходной файл
        for i in range(frames_in_video):
            output_video.write(results[i])

        output_video.release()

        del results

    def test_humans(self):
        return self.get_humans_counter()


if __name__ == '__main__':
    print(f"human = {Labels.human} = {int(Labels.human)}")
    print(f"human = {Labels.helmet} = {int(Labels.helmet)}")
    print(f"uniform = {Labels.uniform} = {int(Labels.uniform)}")
