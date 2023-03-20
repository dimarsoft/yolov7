import numpy as np
import pandas as pd

from count_results import Result, Deviation

"""
tracks это список: [] из [frame_index, track_id, cls, bbox_left (0-1.0), bbox_top(0-1.0), bbox_w(0-1.0), bbox_h(0-1.0), box.conf(0-1.0)]
здесь можно написать свой подсчет и передать функцию
0-1.0 - это условные координаты, а не 640/640
"""


def count_humans(tracks):
    deviations = []
    if len(tracks) == 0:
        count_all = 0
        count_in = 0
        count_out = 0
        return Result(count_all, count_in, count_out, deviations)
    class_hum = 0
    class_helm = 1
    class_uniform = 2
    df = pd.DataFrame(tracks, columns=['frame', 'id', 'cls', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf'])

    # Функция определения индекса первого и последнего индекса появления объекта в видео
    # n - индекс объекта
    def first_last_time(n):
        list = df.index[df.id == n].tolist()
        first = list[0]
        last = list[len(list) - 1]
        return first, last

    # Рассчет количества объектов, пересекших середину кадра по вертикали. '+1' - сверху вниз, '-1' - снизу вверх.
    list_in = []
    count_in = 0
    list_out = []
    count_out = 0
    for n in df.id.unique():  # Проход по всем ид детектированных объектов
        i = first_last_time(n)[0]  # Кадр первого появления объекта с индексом n
        j = first_last_time(n)[1]  # Кадр последнего появления объекта с индексом n
        if df['class'][i] == class_hum:
            if df.bb_top[i] * 640 + df.bb_height[i] * 640 // 2 < 320 and df.bb_top[j] * 640 + df.bb_height[
                j] * 640 // 2 > 320:
                count_in += 1
                list_in.append(n)
            elif df.bb_top[i] * 640 + df.bb_height[i] * 640 // 2 > 320 and df.bb_top[j] * 640 + df.bb_height[
                j] * 640 // 2 < 320:
                count_out += 1
                list_out.append(n)
    # print('Всего вошло человек:', count_in - count_out)
    # print('Список вошедших на территорию', list_in)
    # print('Список вышедших с территории', list_out)
    count_all = count_in + count_out
    # 1-чел без каски и жилета, 2-чел с каской без жилета, 3-чел с жилетом без каски.
    # Получение списка всех ид класса "человек" на видео
    mask = df['class'] == class_hum
    temp = df[mask]
    list_0 = pd.Series(temp.id.unique()).to_list()
    #  print(list_0)
    # Получение списка всех ид класса "жилет" на видео
    mask = df['class'] == class_uniform
    temp = df[mask]
    list_2 = pd.Series(temp.id.unique()).to_list()
    #  print(list_2)
    # Получение списка всех ид класса "каска" на видео
    mask = df['class'] == class_helm
    temp = df[mask]
    list_1 = pd.Series(temp.id.unique()).to_list()

    #  print(list_1)

    # Функция расчета IoU
    class Rectangle:
        def __init__(self, x, y, w, h):
            self.x = x
            self.y = y
            self.w = w
            self.h = h

        def intersection(self, other):
            if ((other.x >= self.x + self.w) or (other.y >= self.y + self.h) or
                    (self.x >= other.x + other.w) or (self.y >= other.y + other.h)):
                return
            #           print('None')
            else:
                R3 = Rectangle(0, 0, 0, 0)
                R3.x = max(self.x, other.x)
                R3.w = min(self.x + self.w, other.x + other.w) - R3.x
                R3.y = max(self.y, other.y)
                R3.h = min(self.y + self.h, other.y + other.h) - R3.y
                return R3

    def IoU(i, j):
        x1 = df.bb_left[i] * 640
        y1 = df.bb_top[i] * 640
        a1 = df.bb_width[i] * 640
        b1 = df.bb_height[i] * 640

        x3 = df.bb_left[j] * 640
        y3 = df.bb_top[j] * 640
        a3 = df.bb_width[j] * 640
        b3 = df.bb_height[j] * 640

        if __name__ == '__main__':
            rect1 = Rectangle(x1, y1, a1, b1)
            rect2 = Rectangle(x3, y3, a3, b3)
            if rect1.intersection(rect2):
                rect3 = rect1.intersection(rect2)
                I = rect3.w * rect3.h
        try:
            U = a1 * b1 + a3 * b3 - I
            IoU = I / U
        except UnboundLocalError:
            IoU = 0
        return IoU

    # Проверка всех вошедших на наличие каски и жилета
    for hum in list_in:
        # Определение значений IoU для всех жилетов
        arr_0_2 = np.full((df.shape[0], len(list_2)), 0.)
        df_0_2 = pd.DataFrame(arr_0_2, columns=list_2)  # Датафрейм с IoU жилетов и людей
        for i in range(df.shape[0]):  # df.shape[0]
            if df.id[i] == hum and df['class'][i] == class_hum:
                frame = df.frame[i]  # Определить кадр
                mask = df.frame == frame
                temp = df[mask]
                for j in range(i, i + temp.shape[0]):  # Проход по кадру - поиск жилета
                    for uniform in list_2:
                        if df.id[j] == uniform and df['class'][
                            j] == class_uniform:  # Определяем жилет. Значение ид берем из list_2
                            df_0_2[uniform][i] = IoU(i, j)
        list_mean_2 = []
        for i in range(len(list_2)):
            list_mean_2.append(df_0_2[list_2[i]].mean())
        try:
            max_value_2 = max(list_mean_2)
            max_index = list_mean_2.index(max_value_2)
            colname_2 = df_0_2.columns[max_index]
        except ValueError:
            #    print('Жилеты отсутствуют')
            max_value_2 = 0
        # Определение значений IoU для всех касок
        arr_0_1 = np.full((df.shape[0], len(list_1)), 0.)
        df_0_1 = pd.DataFrame(arr_0_1, columns=list_1)  # Датафрейм с IoU касок и людей
        for i in range(df.shape[0]):  # df.shape[0]
            if df.id[i] == hum and df['class'][i] == class_hum:
                frame = df.frame[i]  # Определить кадр
                mask = df.frame == frame
                temp1 = df[mask]
                for j in range(i, i + temp1.shape[0]):  # Проход по кадру - поиск каски
                    for helm in list_1:
                        if df.id[j] == helm and df['class'][
                            j] == class_helm:  # Определяем жилет. Значение ид берем из списка list_1
                            df_0_1[helm][i] = IoU(i, j)
        # Определение принадлежности человеку каски
        list_mean_1 = []
        for i in range(len(list_1)):
            list_mean_1.append(df_0_1[list_1[i]].mean())
        try:
            max_value_1 = max(list_mean_1)
            max_index = list_mean_1.index(max_value_1)
            colname_1 = df_0_1.columns[max_index]
        except ValueError:
            #      print('Каски отсутствуют')
            max_value_1 = 0
        if max_value_1 == 0 or max_value_2 == 0:
            #       print('Человеку с id', hum, 'принадлежит каска с id', colname_1, 'и жилет с id', colname_2)
            #       print('Инцидент не выявлен')
            if max_value_1 > 0 and max_value_2 == 0:
                status_id = 2
            #         print('Выявлен инцидент: человек с id', hum, 'не имеет жилета')
            elif max_value_1 == 0 and max_value_2 > 0:
                status_id = 3
            #         print('Выявлен инцидент: человек с id', hum, 'не имеет каски')
            else:
                status_id = 1
            #          print('Выявлен инцидент: человек с id', hum, 'не имеет каски и жилета')
            list = df.index[df.id == hum].tolist()
            #        first = list[0]
            start_frame = df.frame[list[0]]
            #        last = list[len(list)-1]
            end_frame = df.frame[list[len(list) - 1]]

    deviations.append(Deviation(start_frame, end_frame, status_id))

    return Result(count_all, count_in, count_out, deviations)
