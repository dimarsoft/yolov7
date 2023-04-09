# Вариант проверки вошедших и вышедших
import numpy as np
import pandas as pd
import json

from configs import CAMERAS_PATH
from count_results import Result, Deviation


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
            rect = Rectangle(0, 0, 0, 0)
            rect.x = max(self.x, other.x)
            rect.w = min(self.x + self.w, other.x + other.w) - rect.x
            rect.y = max(self.y, other.y)
            rect.h = min(self.y + self.h, other.y + other.h) - rect.y
            return rect


"""tracks это список: [] из [frame_index, track_id, cls, bbox_left (0-1.0), bbox_top(0-1.0), bbox_w(0-1.0), 
bbox_h(0-1.0), box.conf(0-1.0)] здесь можно написать свой подсчет и передать функцию 0-1.0 - это условные координаты, 
а не 640/640"""


# bound_line =  [[490, 662], [907, 613]]
# [[x1, y1], [x2, y2]]
# num(str) - строка = имя файл
# w(int) - ширина
# h(int) - высота

# пример от Александра
def alex_count_humans(tracks, num, w, h, bound_line, log: bool = True):
    if log:
        print(f"Alex post processing v2.3_09.04.2023")
    #    print(f"num = {num}, w = {w}, h = {h}, bound_line = {bound_line}")
    fn = num
    #    print(fn)
    deviations = []
    if len(tracks) == 0:
        count_all = 0
        count_in = 0
        count_out = 0
        return Result(count_all, count_in, count_out, deviations)
    class_hum = 0
    class_helm = 1
    class_uniform = 2
    df = pd.DataFrame(tracks, columns=['frame', 'id', 'class', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf'])
    # Сбрасываем index
    # df.reset_index(drop= True , inplace= True )   # Проверить отключение этой строки
    # Создание датафрейма с траекториями (x,y)
    arr = np.full((df.shape[0], 2), 0.)
    df_tr = pd.DataFrame(arr)
    for i in range(df.shape[0]):
        df_tr[0][i] = int(df.bb_left[i] * 640 + df.bb_width[i] * 640 * 0.3)
        df_tr[1][i] = int(df.bb_top[i] * 640 + df.bb_height[i] * 640 * 0.6)
    df_tr.columns = ['x', 'y']
    # Получение траекторий людей
    mask = df['class'] == class_hum
    temp = df[mask]
    list_0 = pd.Series(temp.id.unique()).to_list()  # Список всех id класса "человек" на видео
    list = []
    for n in list_0:  # цикл по всем людям id [1, 6, 20, 24, 31, 40, 49]
        ind_n = df.index[
            df.id == n].tolist()  # получаем списки индексов где появляется каждый человек [0, 1, 2, 3, 4, 93, 95, 198, 204]
        l = df_tr.values[ind_n].tolist()  # записываем координаты в переменную
        list.append(l)  # добавляем координвиы в список координатных пар точки где появляется человек
    people_tracks = {n: list[list_0.index(n)] for n in list_0}

    # Подсчет вошедших и вышедших (код Тимура)
    def get_proj(p1, p2, m):
        k = (p2[0] - p1[0]) / (p2[1] - p1[1])
        f1 = m[1] + k * m[0]
        f2 = p1[1] - (1 / k) * p1[0]
        x_proj = (f1 - f2) / (k + 1 / k)
        y_proj = f1 - k * x_proj
        return [x_proj, y_proj]

    def get_norm(p1, p2):
        pc = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]
        xmin, xmax = pc[0] - 10, pc[0] + 10
        a = (p2[1] - p1[1]) / (p2[0] - p1[0])
        fnorm = lambda x: pc[1] - (1 / a) * (x - pc[0])
        line_norm = [[xmin, fnorm(xmin)], [xmax, fnorm(xmax)]]
        return line_norm

    # Функция определения пересечения линии турникета
    def crossing_bound(people_path, bound_line):
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        def intersect(A, B, C, D):
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

        if len(people_path) >= 4:
            p1 = [(people_path[0][0] + people_path[1][0]) / 2, (people_path[0][1] + people_path[1][1]) / 2]
            p2 = [(people_path[-2][0] + people_path[-1][0]) / 2, (people_path[-2][1] + people_path[-1][1]) / 2]
        else:
            p1 = [people_path[0][0], people_path[0][1]]
            p2 = [people_path[-1][0], people_path[-1][1]]
        direction = "up" if p2[1] - p1[1] < 0 else "down"
        line_norm = get_norm(*bound_line)
        p1_proj = get_proj(*line_norm, p1)
        p2_proj = get_proj(*line_norm, p2)
        #      a = 20    # Сдвигаем линию турникета на 'a' вниз
        #      bound_line[0][1] = bound_line[0][1] + a
        #      bound_line[1][1] = bound_line[1][1] + a
        intersect = intersect(p1_proj, p2_proj, bound_line[0], bound_line[1])
        return {"direction": direction, "intersect": intersect}

    # Функция подсчета вошедших и вышедших
    def calc_inp_outp_people(tracks_info):
        input_p = 0
        output_p = 0
        for track_i in tracks_info:
            if track_i["intersect"]:
                if track_i["direction"] == 'down':
                    input_p += 1
                elif track_i["direction"] == 'up':
                    output_p += 1
        return {"input": input_p, "output": output_p}

    with open(CAMERAS_PATH, 'r') as f:
        camera_config = json.load(f)

    list_in = []
    list_out = []
    res_inpotp = {}
    bound_line = camera_config.get(fn)

    tracks_info = []
    for p_id in people_tracks.keys():
        people_path = people_tracks[p_id]
        tr_info = crossing_bound(people_path, bound_line)
        if tr_info["intersect"]:
            if tr_info["direction"] == "down":
                list_in.append(p_id)
            else:
                list_out.append(p_id)
        tracks_info.append(tr_info)
    #    print(f"{p_id}: {tr_info}")
    result = calc_inp_outp_people(tracks_info)
    res_inpotp.update({fn: result})
    #    print('Вошли:', list_in, 'Вышли:', list_out)
    #    print(fn, result)
    list_sum = list_in + list_out
    count_in = result["input"]
    count_out = result["output"]
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
    def IoU(i, j):
        x1 = df.bb_left[i] * 640
        y1 = df.bb_top[i] * 640
        a1 = df.bb_width[i] * 640
        b1 = df.bb_height[i] * 640

        x3 = df.bb_left[j] * 640
        y3 = df.bb_top[j] * 640
        a3 = df.bb_width[j] * 640
        b3 = df.bb_height[j] * 640

        rect1 = Rectangle(x1, y1, a1, b1)
        rect2 = Rectangle(x3, y3, a3, b3)
        if rect1.intersection(rect2):
            rect3 = rect1.intersection(rect2)
            rect_i = rect3.w * rect3.h
            try:
                rect_u = a1 * b1 + a3 * b3 - rect_i
                iou = rect_i / rect_u
            except UnboundLocalError:
                iou = 0
            return iou
        else:
            return 0

    # Проверка всех вошедших и вышедших на наличие каски и жилета
    for hum in list_sum:
        # Определение значений IoU для всех жилетов
        arr_0_2 = np.full((df.shape[0], len(list_2)), 0.)
        df_0_2 = pd.DataFrame(arr_0_2, columns=list_2)  # Датафрейм с IoU жилетов и людей
        for i in range(df.shape[0]):  # df.shape[0]
            if df.id[i] == int(hum) and df['class'][i] == class_hum:
                frame = df.frame[i]  # Определить кадр
                mask = df.frame == frame
                temp = df[mask]
                for j in range(i, i + temp.shape[0]):  # Проход по кадру - поиск жилета
                    for uniform in list_2:
                        # Определяем жилет. Значение ид берем из list_2
                        if df.id[j] == uniform and df['class'][j] == class_uniform:
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
            if df.id[i] == int(hum) and df['class'][i] == class_hum:
                frame = df.frame[i]  # Определить кадр
                mask = df.frame == frame
                temp1 = df[mask]
                for j in range(i, i + temp1.shape[0]):  # Проход по кадру - поиск каски
                    for helm in list_1:
                        # Определяем жилет. Значение ид берем из списка list_1
                        if df.id[j] == helm and df['class'][j] == class_helm:
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
            list_hum = df.index[df.id == hum].tolist()
            #        first = list_hum[0]
            start_frame = df.frame[list_hum[0]]
            #        last = list[len(list_hum)-1]
            end_frame = df.frame[list_hum[len(list_hum) - 1]]

            deviations.append(Deviation(int(start_frame), int(end_frame), int(status_id)))

    return Result(int(count_all), int(count_in), int(count_out), deviations)
