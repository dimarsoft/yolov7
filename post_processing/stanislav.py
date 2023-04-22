import pandas as pd
from operator import attrgetter
from pandas import DataFrame

from tools.count_results import Result
from tools.exception_tools import print_exception


# Классы
class Human():
    def __init__(self):
        # self.frame = -1
        self.id = -1
        self.z = []  # список по кадрам зона входа или выхода (1 или 2)
        self.m = 0  # результат - вошел или вышел
        self.k = 0


# ====================
class Box():
    def __init__(self):
        self.id = -1
        self.l = 0
        self.t = 0
        self.r = 0
        self.b = 0
        self.cl = -1

        self.pk = []
        self.pu = []
        # self.ks = []
        self.mpk = 0.0
        self.mpu = 0.0
        self.k = -1
        self.u = -1
        # self.k_b = 0


class Frame():
    def __init__(self):
        self.fr = -1
        self.boxs = []

        self.hb = []
        self.kb = []
        self.ub = []


class PassCouter():  #
    def __init__(self):
        self.id = -1
        self.inc = 0
        self.out = 0

        self.frc = 0
        self.kc = 0
        self.uc = 0
        # self.k_b = 0


# Функции учета касок на людях
def PartKaski(h, k):
    l = max([h[0], k[0]])
    t = max([h[1], k[1]])
    r = min([h[2], k[2]])
    b = min([h[3], k[3]])
    area_k = (k[2] - k[0]) * (k[3] - k[1])
    area_i = (r - l) * (b - t)
    if area_i < 0:
        area_i = 0
    return area_i / area_k


def Frame_f(data, passlist):
    # перенос информации из датафрейма в список объектов класса Frame (Framelist),
    # включающий в себя класс Box
    # объект Frame - поля - номере кадра, список объектов класса Box
    # объект Box - поля - id бокса, параметры бокса (left,top,right,bottom), класс бокса

    Framelist = []
    # разбиваем все строки по кадрам.
    # боксы строк с одним и тем же фреймом записываем в лист (f.boxs)
    for i in range(len(data)):
        # перенос данных из датафрейма в класс Frame. Список  кадров  объектов Frame (Framelist).
        box = Box()
        box.id = data.iloc[i].id
        box.l = data.iloc[i].left
        box.t = data.iloc[i].top
        box.r = data.iloc[i].width + data.iloc[i].left
        box.b = data.iloc[i].height + data.iloc[i].top
        box.cl = data.iloc[i].cl

        if i != 0:
            if (data.iloc[i].frame != data.iloc[
                i - 1].frame):  # новый кадр (для исключения строк с одними и теми же кадрами но с разными боксами)

                f = Frame()  # создаем объект кадра
                f.fr = data.iloc[i].frame
                f.boxs.append(box)

                Framelist.append(f)
            else:
                # f.fr = data.iloc[i].frame
                f.boxs.append(box)

        if i == 0:  # первый кадр и первая строка
            f = Frame()
            f.fr = data.iloc[i].frame
            f.boxs.append(box)

            Framelist.append(f)

    # В результате получили лист объектов кадров Framelist. Для каждого объекта кадра заполнены данные о номере кадра (.fr) и список боксов .boxs
    # ========================================================
    # ищем соответствие бокса каски боксу человека

    for f in Framelist:  # перебор кадров
        for i in f.boxs:
            if i.cl == 0:
                f.hb.append(i)  # список всех id людей в кадре
            if i.cl == 1:
                f.kb.append(i)  # список всех касок в кадре
            if i.cl == 2:
                f.ub.append(i)  # список всех жилетов в кадре

        for h in f.hb:
            hl = [h.l, h.t, h.r, h.b]
            for k in f.kb:
                kl = [k.l, k.t, k.r, k.b]
                pk = PartKaski(hl, kl)  # вероятность принадлежности бокса каски k к h (человеку)

                if pk > 1:
                    pk = 0
                h.pk.append(pk)  # сохраняем в классе

            for u in f.ub:
                ul = [u.l, u.t, u.r, u.b]
                pu = PartKaski(hl, ul)  # вероятность принадлежности бокса каски k к h (человеку)

                if pu > 1:
                    pu = 0
                h.pu.append(pu)  # сохраняем в классе
    # -----------------------------------

    for f in Framelist:
        # максимальное соответствие бокса человека каске
        for h in f.hb:  # f.hb - список боксов людей в кадре
            if len(h.pk) != 0:
                mp = max(h.pk)
                h.mpk = mp
                index_max = h.pk.index(mp)
                h.k = f.kb[index_max].id  # максимальное соответствие бокса человека каске

            if len(h.pu) != 0:
                mp = max(h.pu)
                h.mpu = mp
                index_max = h.pu.index(mp)
                h.u = f.ub[index_max].id  # максимальное соответствие бокса человека жилету

        # ========================================================
        for k in f.kb:  # f.kb - список боксов касок в кадре
            hls = []
            for h in f.hb:
                if k.id == h.k:
                    hls.append(h)
            max_attr = h
            if (len(hls) > 1):
                max_attr = max(hls, key=attrgetter('mpk'))

            for h in f.hb:
                if (max_attr.id == h.id):
                    h.k = max_attr.k
                else:
                    h.k = -1

        for u in f.ub:  # f.ub - список боксов жилетов в кадре
            hls = []
            for h in f.hb:
                if u.id == h.u:
                    hls.append(h)
            max_attr = h
            if (len(hls) > 1):
                max_attr = max(hls, key=attrgetter('mpu'))

            for h in f.hb:
                if (max_attr.id == h.id):
                    h.u = max_attr.u
                else:
                    h.u = -1

    for f in Framelist:
        for i in f.boxs:
            if i.cl == 0:
                for pl in passlist:
                    if pl.id == i.id:
                        pl.frc += 1
                        if i.k != -1:
                            pl.kc += 1

                        if i.u != -1:
                            pl.uc += 1

    return passlist


# Разбиение датафрейма на id
def SplitDf(data):
    cls0 = data[data['cl'] == 0]
    df_l = []
    if len(cls0) > 0:
        for i in range(cls0['id'].min(), cls0['id'].max() + 1):  # от min id до max id
            if len(cls0[cls0['id'] == i]) != 0:  # выбор существующих в ролике id
                tmp_h = cls0[cls0['id'] == i]  # выборка датафрейма с конкретным id человека

                begin = 1
                for q in range(1, len(tmp_h)):
                    if q == (len(tmp_h) - 1):
                        df_l.append(tmp_h[begin:(q - 1)])
                    if (tmp_h.iloc[q].frame - tmp_h.iloc[q - 1].frame) > 20:
                        df_l.append(tmp_h[begin:(q - 1)])
                        begin = q

    return df_l


# Функции счетчика прошедших через турникет людей

def Human_f(Y, data):
    # ---------------------------------------------
    r_in = r_out = 0  # переменные для подсчета входящих и выходящих
    humanIdList = []  # список id людей в ролике
    df_l = SplitDf(data)

    for i in range(len(df_l)):
        h = Human();
        h.id = i;
        y_b_0 = None;  # создание объекта класса

        for q in range(len(df_l[i])):
            y_b = df_l[i].iloc[q].top + df_l[i].iloc[q].height  # (y) нижнего края бокса

            # ----  определение стартовой позиции бокса  ---------------
            if y_b_0 == None:
                y_b_0 = y_b
                if y_b < Y:  # Y - y турникета
                    h.z.append(1)  # 1 = зона входа
                if y_b > Y:
                    h.z.append(2)  # 2 = зона выхода

            # ----- определение остальных позиций бокса (в остальных кадрах)-----

            if (y_b < Y) & (h.z[-1] == 2):
                h.z.append(1)  # 1 = зона входа
            if (y_b > Y) & (h.z[-1] == 1):
                h.z.append(2)  # 2 = зона выхода

        humanIdList.append(h)

    # -----------------------------------------------------------
    for j in humanIdList:  # определение наличия прохода через турникет и их подсчет

        if len(j.z) > 1:
            if j.z[0] > j.z[-1]:  # старторая зона j.z[0] больше последней зоны j.z[-1] - 2>1 - это выход
                j.m = 2  # фиксация выхода для id
                r_out += 1
            if j.z[0] < j.z[-1]:  # старторая зона j.z[0] меньше последней зоны j.z[-1] - 1<2 - это вход
                j.m = 1  # фиксация входа для id
                r_in += 1

    PassList = []  # лист объектов PassCouter (только прошедших через турникет id ) нужен для определения нарушений
    for j in humanIdList:

        if j.m != 0:
            pc = PassCouter()
            pc.id = j.id

            if j.m == 1:
                pc.inc = 1
                PassList.append(pc)
            if j.m == 2:
                pc.out = 1
                PassList.append(pc)

    return r_in, r_out, PassList  # колическтво входов и выходов, список id прошедщих турникет


def END(p, f, turniket_dict, df):  # работа с файлом полученного трекером

    path = '/content/drive/MyDrive/yolov8_tracking/runs/' + p + str(
        f) + '.txt'  # путь к конктетному файлу .txt выхода трекера по конкретному видео
    # конвертация файла в датафрейм
    try:  # если есть такой файл то обрабатываем
        # создаем датафрейм для переноса данных из .txt файла
        tracker_data = pd.read_csv(path, sep=' ', header=None, usecols=[0, 1, 2, 3, 4, 5, 6])
        tracker_data.columns = ['frame', 'id', 'left', 'top', 'width', 'height', 'cl']

        # подсчет входов и выходов  (людей)
        r_in, r_out, passlist = Human_f(turniket_dict[f], tracker_data)
        '''
        plist = Frame_f(tracker_data,passlist)
        k = 0; u = 0
        for i in plist:
          if i.kc/i.frc > 0.25:
            k+=1
          if i.uc/i.frc > 0.25:
            u+=1
        '''
        df.loc[f] = [f, r_in, r_out, 0, 0]  # кадр, in, out, каски, жилеты
        # df.loc[f] = [f,r_in,r_out,k,u]    # кадр, in, out, каски, жилеты
    except:  # если .txt файла нет (в пустых видео) , тогда нули.
        df.loc[f] = [f, 0, 0, 0, 0]


def convert_track_to_df(tracks: list, w, h) -> DataFrame:
    """
    Конвертация в DataFrame нужный постобработке
    Args:
        tracks: Список треков
        w: Ширина фрейма(картинки)
        h: Высота

    Returns:
        DataFrame(columns=['frame', 'id', 'left', 'top', 'width', 'height', 'cl']
    """
    new_track = []
    # [frame_index, track_id, cls, bbox_left, bbox_top, bbox_w, bbox_h, box.conf]

    for item in tracks:
        cls = item[2]
        bbox_left, bbox_top, bbox_w, bbox_h = item[3] * w, item[4] * h, item[5] * w, item[6] * h

        new_item = [int(item[0]), int(item[1]), int(bbox_left), int(bbox_top), int(bbox_w), int(bbox_h), int(cls)]

        new_track.append(new_item)

    tracker_data = DataFrame(new_track, columns=['frame', 'id', 'left', 'top', 'width', 'height', 'cl'])

    return tracker_data


# пример от Станислава
def stanislav_count_humans(tracks: list, num, w, h, bound_line, log: bool = True) -> Result:
    # Турникет Станислава, потом нужно перейти на общий

    turniket_dict: dict[int, int] = \
        {1: 470, 2: 470, 3: 470, 4: 910, 5: 470,
         6: 580, 7: 580, 8: 470, 9: 470, 10: 470,
         11: 470, 12: 470, 13: 470, 14: 580, 15: 470,
         16: 470, 17: 470, 18: 470, 19: 470, 20: 850,
         21: 430, 22: 430, 23: 430, 24: 430, 25: 510,
         26: 510, 27: 510, 28: 510, 29: 510, 30: 510,
         31: 430, 32: 0, 33: 0, 34: 0, 35: 0,
         36: 430, 37: 0, 38: 430, 39: 430, 40: 430,
         41: 0, 42: 430, 43: 0
         }
    f = int(num)

    turnic_y = turniket_dict.get(f)

    if turnic_y is None:
        pt1 = bound_line[0]
        pt2 = bound_line[1]
        turnic_y = int((pt1[1] + pt2[1]) / 2)

    tracker_data = convert_track_to_df(tracks, w, h)

    try:
        # подсчет входов и выходов (людей)
        r_in, r_out, passlist = Human_f(turnic_y, tracker_data)
        return Result(r_in + r_out, r_in, r_out, [])
    except Exception as ex:
        print_exception(ex, "Human_f")

    # ошибка будет в print_exception, но результат вернем
    return Result(0, 0, 0, [])
