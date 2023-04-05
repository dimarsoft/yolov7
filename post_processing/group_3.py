import pandas as pd
from pandas import DataFrame

from count_results import Result, Deviation


def convert_tracks_df(tracks, w, h) -> DataFrame:
    new_track = []
    # [frame_index, track_id, cls, bbox_left, bbox_top, bbox_w, bbox_h, box.conf]

    for item in tracks:
        cls = item[2]
        conf = item[7]
        bbox_left, bbox_top, bbox_w, bbox_h = item[3] * w, item[4] * h, item[5] * w, item[6] * h

        cntrx = bbox_left + bbox_w / 2
        cntry = bbox_top + bbox_h / 2

        new_item = [item[0], item[1], cls, conf, bbox_left, bbox_top, bbox_left + bbox_w, bbox_top + bbox_h, cntrx, cntry]

        new_track.append(new_item)

    df = DataFrame(new_track,
                   columns=["frame", "id", "class", "confidence", "bb_left", "bb_top", "bb_right", "bb_bottom", "cntrx",
                            "cntry"])

    return df


def convert_tracks(tracks, w, h) -> list:
    new_track = []
    # [frame_index, track_id, cls, bbox_left, bbox_top, bbox_w, bbox_h, box.conf]

    for item in tracks:
        if item[2] != 0:
            continue
        bbox_left, bbox_top, bbox_w, bbox_h = item[3], item[4], item[5], item[6]
        new_item = [item[0], item[1], bbox_left * w, bbox_top * h, bbox_w * w, bbox_h * h]

        new_track.append(new_item)

    return new_track


def calc_inp_outp_people(tracks, width, height):
    gate_y = 375 * height // 640  # примерная координата турникета
    delta = 50 * height // 640  # эмпирическая поправка (она вроде не особо влияет, но пусть будет)
    gate_y -= delta
    res = height  # размер видео по вертикали

    n_humans = 0  # общее количество id в треке (далее может увеличиться)

    for t in tracks:
        n_humans = (max(t[1], n_humans))

    '''
    Предобработка треков.
    Иногда один и тот же id сначала проходит в одном направлении, а потом в другом, нужно посчитать его как одного входящего и одного выходящего.
    В данной реализации сравнивается первая и последняя координата объекта, поэтому если он сначала вошёл, а потом вышел, начальная и последняя координаты будут по одну сторону от турникета и алгоритм не сработает.
    Предпринята попытка форсированной смены id. Идея: после того как объект прошёл коридор - все дальнейшие id этого объекта заменяются на новые, и при последующем проходе, он будет считаться как новый объект с новым id.
    '''

    bb_y = [None] + [list() for _ in range(n_humans)]  # координата 'у' верхнего левого угла
    bb_h = [None] + [list() for _ in range(n_humans)]  # высота баундин бокса
    bb_frame = [None] + [list() for _ in range(n_humans)]  # номер кадра
    bb_start = [None] + [None for _ in
                         range(n_humans)]  # с какой стороны заходит (= 0, если чел выходит,  = res, если заходит)

    for i, t in enumerate(tracks):
        id = t[1]
        frame_n = t[0]
        x, y, w, h = t[2:]
        bb_y[id].append(y)
        bb_h[id].append(h)
        bb_frame[id].append(frame_n)
        if i == 0:
            bb_start[id] = bb_y[id][0]
        else:
            if bb_y[id][-1] < bb_y[id][0]:
                bb_start[id] = res
            else:
                bb_start[id] = 0

        if len(bb_frame[id]) > 20:

            condition_in = bb_start[id] == 0 and (bb_y[id][-1] + bb_h[id][-1]) / res > 0.75 and bb_frame[id][-1] - \
                           bb_frame[id][
                               -2] > 20  # условие смены id при проходе внутрь: 1) стартовая точка наверху, 2) нижняя граница бб пересекла порог (0,75 от высоты изображения) 3) бб не детектировался в течение 10 предыдущих кадров                                         # go in
            condition_out = bb_start[id] == res and bb_y[id][-1] / res < 0.35 and bb_frame[id][-1] - bb_frame[id][
                -2] > 20  # условие смены id при проходе наружу: 1) стартовая точка внизу, 2) верхняя граница бб пересекла порог (0,35 от высоты изображения) 3) бб не детектировался в течение 10 предыдущих кадров

            if condition_in or condition_out:
                # print(f'New human {id} -> {n_humans+1}, frame: {frame_n}')
                # print()
                n_humans += 1
                bb_y.append([])
                bb_h.append([])
                bb_frame.append([])
                bb_start.append(None)
                for j in range(i + 1, len(tracks)):  # во всех последующих детекциях данный id меняются новый
                    if tracks[j][1] == id:
                        tracks[j][1] = n_humans

    '''
    Собственно подсчёт.
    Если условная координата "y" турникета находится в интервале между начальной и конечной координатой объекта, то защитывается проход
    '''

    came_in = 0  # количество вошедших
    came_out = 0  # количество вышедших

    human_tracks = [None] + [list() for _ in
                             range(n_humans)]  # нулевой элемент добавлен, чтобы нумерация начиналась с 1

    # print(f'Track {N}):')

    for t in tracks:
        n = t[1]
        x, y, w, h = t[2:]
        box_center = x + w // 2, y + h // 2
        human_tracks[n].append(box_center[1])

    for i, h in enumerate(human_tracks):
        if h and h[0] < gate_y < h[-1]:
            came_in += 1
            # print(f'          Human {i} came in')

        elif h and h[0] > gate_y > h[-1]:
            came_out += 1

    return came_in, came_out


# Задаем функцию who_came_left, принимает датафрейм, возвращает id людей, которые вошли и вышли.
def who_came_left(data):
    frame_top = data['bb_top'].min()  # кадры иногда имеют черную рамку, ищем границы картинки
    frame_bottom = data['bb_bottom'].max()
    ppl_who_came = []
    ppl_who_left = []
    for idx, df in data.groupby(['class', 'id']):
        if idx[0] == 0:  # only for human class
            print(idx)
            cntry_min = df['cntry'].min()  # человек иногда заходит и сразу выходит
            cntry_max = df['cntry'].max()  # поэтому учитываем не только его начальную и конечную координаты,
            cntry_start = df['cntry'].iat[0]  # но и его минимальную и максимальную координаты.
            cntry_end = df['cntry'].iat[-1]
            print(f'cntry_min = {cntry_min}, cntry_max = {cntry_max}')
            print(f'cntry_start = {cntry_start}, cntry_end = {cntry_end}')
            moved = (cntry_max - cntry_min) / (frame_bottom - frame_top)
            print(f'Human {idx[1]} moved {int(100 * moved)}% of the frame (vertically)')
            if moved >= 0.25:  # если центр бб человека переместился более 25% кадра по вертикали, считаем прошедшим
                if (
                        0.8 * cntry_min < cntry_start < 1.2 * cntry_min and  # если человек начал путь примерно наверху кадра
                        0.8 * cntry_max < cntry_end < 1.2 * cntry_max):  # если человек закончил путь примерно внизу кадра
                    ppl_who_came.append(idx[1])  # то мы засчитываем его как вошедшего
                    print(f'Human {idx[1]} came in.')
                elif (
                        0.8 * cntry_min < cntry_end < 1.2 * cntry_min and  # если человек закончил путь примерно наверху кадра
                        0.8 * cntry_max < cntry_start < 1.2 * cntry_max):  # если человек начал путь примерно внизу кадра
                    ppl_who_left.append(idx[1])  # то мы засчитываем его как вышедшего
                    print(f'Human {idx[1]} left.')
                elif (
                        0.8 * cntry_min < cntry_start < 1.2 * cntry_min and  # если человек начал путь примерно наверху кадра
                        0.8 * cntry_min < cntry_end < 1.2 * cntry_min):  # если человек закончил путь примерно наверху кадра
                    ppl_who_came.append(idx[1])  # то мы считаем, что он вошел и сразу вышел
                    ppl_who_left.append(idx[1])
                    print(f'Human {idx[1]} came and left.')
                elif (
                        0.8 * cntry_max < cntry_start < 1.2 * cntry_max and  # если человек начал путь примерно внизу кадра
                        0.8 * cntry_max < cntry_end < 1.2 * cntry_max):  # если человек закончил путь примерно внизу кадра
                    ppl_who_came.append(idx[1])  # то мы считаем, что он вышел и сразу вернулся
                    ppl_who_left.append(idx[1])
                    print(f'Human {idx[1]} left and came back.')
                else:
                    print(f'Can not define what human {idx[1]} did.')
    return ppl_who_came, ppl_who_left


# Зададим функцию get_who_wears_what,
# на вход - часть датафрейма с инфо по одному кадру
# на выход - список кортежей, где каждый кортеж это:
# (id человека, True/False на наличие каски, True/False на наличие жилета)
def get_who_wears_what(current_frame):
    try:
        humans_frame = current_frame.loc[0]
        humans = True
        if isinstance(humans_frame, pd.Series):
            humans_frame = humans_frame.to_frame().transpose()
    except:
        # print(f"There are no humans in frame {frame}.")
        humans = False
    try:
        hardhats_frame = current_frame.loc[1]
        hardhats = True
        if isinstance(hardhats_frame, pd.Series):
            hardhats_frame = hardhats_frame.to_frame().transpose()
    except:
        # print(f"There are no hardhats in frame {frame}.")
        hardhats = False
    try:
        vests_frame = current_frame.loc[2]
        vests = True
        if isinstance(vests_frame, pd.Series):
            vests_frame = vests_frame.to_frame().transpose()
    except:
        # print(f"There are no vests in frame {frame}.")
        vests = False
    frame_info = []
    if humans:
        for index, row in humans_frame.iterrows():
            current_human = [int(row['id']), False, False]
            if hardhats:
                for idx, rw in hardhats_frame.iterrows():
                    if row['bb_left'] < rw['cntrx'] < row['bb_right'] and row['bb_top'] < rw['cntry'] < row[
                        'bb_bottom']:
                        # print(f"human {row['id']} has a hat")
                        current_human[1] = True
            if vests:
                for idx, rw in vests_frame.iterrows():
                    if row['bb_left'] < rw['cntrx'] < row['bb_right'] and row['bb_top'] < rw['cntry'] < row[
                        'bb_bottom']:
                        # print(f"human {row['id']} has a vest")
                        current_human[2] = True
            current_human = tuple(current_human)
            # print(current_human)
            frame_info.append(current_human)
    # print(frame_info)
    return frame_info


# Теперь, когда у нас есть словарь info, мы можем оценить в каком количестве кадров у человека детектится каска и жилет
def get_good_bad_human(id, info, fps):
    human_frames = []
    hardhat_frames = []
    vest_frames = []
    for key, value in info.items():
        for human in value:
            if human[0] == id:
                human_frames.append(key)
                if human[1] == True:
                    hardhat_frames.append(key)
                if human[2] == True:
                    vest_frames.append(key)
    start_frame = min(human_frames)
    end_frame = max(human_frames)
    start_time = int(start_frame // fps)
    end_time = int(end_frame // fps)
    human_frames_n = len(human_frames)
    hardhat_frames_n = len(hardhat_frames)
    vest_frames_n = len(vest_frames)
    violations_dict = {0: 'all good!', 1: 'no hat, no vest', 2: 'no vest', 3: 'no hat'}
    print(
        f'Human {id} is detected in {human_frames_n} frames, we detect that he has a hat in {hardhat_frames_n} frames and vest in {vest_frames_n} frames.')
    if hardhat_frames_n / human_frames_n >= 0.7 and vest_frames_n / human_frames_n >= 0.7:
        violation_key = 0
        print(f"Human {id} is a good human. :-)")
    elif hardhat_frames_n / human_frames_n >= 0.7:
        violation_key = 2
        print(
            f'Human with id {id} is crossing with {violations_dict[violation_key]} in interval {start_time // 60} min {start_time % 60} sec - {end_time // 60} min {end_time % 60} sec.')
    elif vest_frames_n / human_frames_n >= 0.7:
        violation_key = 3
        print(
            f'Human with id {id} is crossing with {violations_dict[violation_key]} in interval {start_time // 60} min {start_time % 60} sec - {end_time // 60} min {end_time % 60} sec.')
    else:
        violation_key = 1
        print(
            f'Human with id {id} is crossing with {violations_dict[violation_key]} in interval {start_time // 60} min {start_time % 60} sec - {end_time // 60} min {end_time % 60} sec.')
    return id, violation_key, start_time, end_time, start_frame, end_frame


def get_deviations(tracks, w, h, fps):
    data = convert_tracks_df(tracks, w, h)

    # Сортируем по значениям кадров и классов (по возрастанию)
    data.sort_values(['frame', 'class'], ascending=[True, True], inplace=True)

    # Создаем мультииндекс
    data.set_index(['frame', 'class'], inplace=True)

    # Применим функцию who_came_left и получим списки id прошедших людей.
    ppl_who_came, ppl_who_left = who_came_left(data)

    # Используем созданную функцию get_who_wears_what на данных
    # Получаем словарь info:
    # ключи - номера кадров,
    # значения - списки кортежей, где каждый кортеж это:
    # (id человека, True/False на наличие каски, True/False на наличие жилета)
    info = {}
    for frame, new_df in data.groupby(level=0):
        current_frame = new_df.droplevel(0)
        frame_info = get_who_wears_what(current_frame)
        info[frame] = frame_info

    devs = []
    # Применим функцию get_good_bad_human к полученным ранее спискам вошедших и вышедших
    for track_id in ppl_who_came:
        track_id, violation_key, start_time, end_time, start_frame, end_frame = get_good_bad_human(track_id, info, fps=fps)

        if violation_key > 0:  # 0 нет нарушения
            devs.append(Deviation(start_frame, end_frame, violation_key))

    return devs


def group_3_count(tracks, num, w, h, fps):
    print(f"Group 3 post processing v1.2 (04.04.2023)")

    new_tracks = convert_tracks(tracks, w, h)

    came_in, came_out = calc_inp_outp_people(new_tracks, w, h)

    deviations = get_deviations(tracks, w, h, fps)

    print(f"{num}: count_in = {came_in}, count_out = {came_out}, deviations = {len(deviations)}")

    return Result(came_in + came_out, came_in, came_out, deviations)
