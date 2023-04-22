"""

Файл для описания классов результатов детекции и трекинга.
Удобно объекты классов потом сериализовать(сохранить) в json формат

"""
from labeltools import Labels


class BBox:
    def __init__(self, bbox):
        """

        Args:
            bbox (): Массив:
            0 left
            1 top
            2 width
            3 height
        """
        self.left = float(bbox[0])
        self.top = float(bbox[1])
        self.width = float(bbox[2])
        self.height = float(bbox[3])


class Detection:
    def __init__(self, bbox, cls, conf, frame_index):
        """

        Args:
            bbox : bbox
            cls : класс
            conf : conf
            frame_index : индекс фрейма

        """
        self.bbox = BBox(bbox)
        self.cls = int(cls)
        self.conf = float(conf)
        self.frame_index = int(frame_index)


class Track(Detection):
    def __init__(self, bbox, cls, conf, frame_index, track_id):
        """

        Args:
            bbox : bbox
            cls : класс
            conf : conf
            frame_index : индекс фрейма
            track_id : ИД трека
        """
        super().__init__(bbox, cls, conf, frame_index)

        self.track_id = int(track_id)


def get_classes() -> list[int]:
    """
    Классы объектов, с которыми работаем
    Returns:
        Список классов

    """
    return [int(Labels.human), int(Labels.helmet), int(Labels.uniform)]
