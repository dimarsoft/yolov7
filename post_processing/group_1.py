import numpy as np
from numpy import ndarray

from post_processing.group_1_tools import get_men, get_count_men, get_count_vialotion
from tools.count_results import Result, Deviation
from tools.labeltools import get_status


def dict_fill(d_pred_, d_true_):
    # Функция, которая выравнивает содержимое в предсказанном словаре с эталоном, заполняя его 0
    d_pred_exp = d_pred_.copy()  # создаем новые словари
    d_true_exp = d_true_.copy()

    # если длина значений словарей не совпадает делаем сортировку от 1 к 0 и добиваем значения словаря меньшео
    # размера маркером отсутсвия детекции - 2
    if len(d_pred_exp) != len(d_true_exp):
        d_pred_exp.sort(reverse=True)
        d_true_exp.sort(reverse=True)
        _list = [d_pred_exp, d_true_exp]
        for a in _list:
            a.extend([2] * (max(map(len, _list)) - len(a)))

    return d_pred_exp, d_true_exp


def demo():
    # Демонстрация!
    # 1 - одного одетого задетектил как двоих раздетых
    # 2 - лишние треки в середине видео
    # 3 - пропуск отсутсвие детекции в конце
    # 4 - объеденил в середине

    d_pred = {'1': [1, 0, 0], '2': [1, 0, 0, 0, 1, 0], '3': [1, 0], '4': [1, 0, 1]}
    d_true = {'1': [1, 1], '2': [1, 0, 1, 0], '3': [1, 0, 0, 1], '4': [1, 0, 0, 1]}
    for i in range(1, 5):
        d_pred[f'{i}'], d_true[f'{i}'] = dict_fill(d_pred[f'{i}'], d_true[f'{i}'])
    print(d_pred)
    print(d_true)


def convert_track_to_ny(tracks: list, w, h) -> ndarray:
    # bboxes - ndarray(x1, y1, x2, y2, conf, class, id, frame),
    """
    Конвертация в DataFrame нужный постобработке
    Args:
        tracks: Список треков
        w: Ширина фрейма(картинки)
        h: Высота

    Returns:
        bboxes - ndarray(x1, y1, x2, y2, conf, class, id, frame),
    """
    new_track = []
    # [frame_index, track_id, cls, bbox_left, bbox_top, bbox_w, bbox_h, box.conf]

    for item in tracks:
        cls = int(item[2])
        conf = item[7]
        frame_index, track_id = int(item[0]), int(item[1])
        bbox_left, bbox_top, bbox_w, bbox_h = item[3] * w, item[4] * h, item[5] * w, item[6] * h

        new_item = [int(bbox_left), int(bbox_top), int(bbox_left + bbox_w), int(bbox_top + bbox_h),
                    conf, int(cls), track_id, frame_index]

        new_track.append(new_item)

    return np.asarray(new_track)


# Постобработка Группа №1
def group_1_count_humans(tracks: list, num, w, h, bound_line, log: bool = True) -> Result:

    if len(tracks) == 0:
        return Result(0, 0, 0, [])

    orig_shp = [w, h]

    out_boxes = convert_track_to_ny(tracks, w, h)
    # Смотрим у какого айди есть каски и жилеты (по порогу от доли кадров
    # где был зафиксирован айди человека + каска и жилет в его бб и без них)
    men = get_men(out_boxes)

    # здесь переназначаем айди входящий/выходящий (временное решение для MVP, надо думать над продом)
    men_clean, incoming1, exiting1 = get_count_men(men, orig_shp[1])

    # Здесь принимаем переназначенные айди смотрим нарушения, а также повторно считаем входящих по дистанции, проверяем
    violation, incoming2, exiting2, df, clothing_helmet, clothing_unif = get_count_vialotion(men_clean, orig_shp[1])

    deviations = []

    # 'helmet', 'uniform', 'first_frame', 'last_frame'

    for row in range(len(violation)):
        start_frame = violation["first_frame"].iloc[row]
        end_frame = violation['last_frame'].iloc[row]

        helmet = violation["helmet"].iloc[row]
        uniform = violation['uniform'].iloc[row]

        status = get_status(helmet == 1, uniform == 1)
        deviations.append(Deviation(int(start_frame), int(end_frame), status))

    return Result(incoming2 + exiting2, incoming2, exiting2, deviations)


if __name__ == '__main__':
    demo()
