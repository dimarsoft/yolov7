from exception_tools import print_exception


def scale_bbox(bbox, scale: float):
    """

    Args:
        bbox: bbox на замену
        scale (float):
    """
    x1_center = (bbox[:, [0]] + bbox[:, [2]]) / 2
    y1_center = (bbox[:, [1]] + bbox[:, [3]]) / 2

    scale /= 2

    w = abs(bbox[:, [0]] - bbox[:, [2]]) * scale
    h = abs(bbox[:, [1]] - bbox[:, [3]]) * scale

    bbox[:, [0]] = x1_center - w
    bbox[:, [2]] = x1_center + w

    bbox[:, [1]] = y1_center - h
    bbox[:, [3]] = y1_center + h

    return bbox


def change_bbox(bbox, change_bb, file_id=None, clone=False):
    if change_bb is None:
        return bbox

    if callable(change_bb):
        try:
            if clone:
                bbox = bbox.clone()
            return change_bb(bbox, file_id)
        except Exception as ex:
            print_exception(ex, "external change bbox")
            return bbox

    if isinstance(change_bb, float):
        if clone:
            bbox = bbox.clone()
        return scale_bbox(bbox, change_bb)

    if not isinstance(change_bb, bool):
        return bbox

    if not change_bb:
        return bbox

    if clone:
        bbox = bbox.clone()

    x1 = (bbox[:, [0]] + bbox[:, [2]]) / 2
    y1 = (bbox[:, [1]] + bbox[:, [3]]) / 2

    w = 10  # abs(bbox[:, [0]] - bbox[:, [2]]) / 4
    h = 10  # abs(bbox[:, [1]] - bbox[:, [3]]) / 4

    bbox[:, [0]] = x1 - w
    bbox[:, [2]] = x1 + w

    bbox[:, [1]] = y1 - h
    bbox[:, [3]] = y1 + h

    return bbox


def no_change_bbox(bbox, file_id: str):
    """

    Args:
        file_id(str): имя файла
        bbox (tensor): первые 4ре столбца x1 y1 x2 y2
        в абсолютных значениях картинки.
        Пока это ббоксы всех классов,
        сам класс находится в последнем столбце (-1, или индекс 5)
    """
    print(f"file = {file_id}")
    return bbox


# пример реализации бокса по центру 20/20
def change_bbox_to_center(bbox, file_id):
    x1 = (bbox[:, [0]] + bbox[:, [2]]) / 2
    y1 = (bbox[:, [1]] + bbox[:, [3]]) / 2

    w = 10
    h = 10

    bbox[:, [0]] = x1 - w
    bbox[:, [2]] = x1 + w

    bbox[:, [1]] = y1 - h
    bbox[:, [3]] = y1 + h

    return bbox


# пример реализации от Павла (группа №1)
def pavel_change_bbox(bbox, file_id):
    y2 = bbox[:, [1]] + 150

    bbox[:, [3]] = y2

    return bbox
