from ultralytics import YOLO
import numpy as np
import pandas as pd
from IPython.display import clear_output
import cv2
import os
from sklearn.metrics import precision_recall_fscore_support


def get_boxes(
        result):  # эта функция сохраняет боксы от предикта в файл .npy для того что бы не возвращаться больше к детекции
    orig_shp = result[0].orig_shape
    all_boxes = np.empty((0, 7))
    for i in range(len(result)):
        bbox = result[i].cpu().boxes.data.numpy()
        bbox = np.hstack((bbox, np.tile(i, (bbox.shape[0], 1))))
        all_boxes = np.vstack((all_boxes, bbox))
    return result, all_boxes, orig_shp


def detect_videos(path_model, model_in_path, video_source, start_vid=1, end_vid=1):
    if end_vid == 1:
        length = len([f for f in os.listdir(video_source)
                      if f.endswith('.mp4') and os.path.isfile(
                os.path.join(video_source, f))])  # подсчитаем количество видео в папке
    else:
        length = end_vid
    for N in range(start_vid, length + 1):  # устанавливаем какие видео смотрим
        try:
            with open(video_source + f'{N}.mp4', 'r') as f:
                model = YOLO(
                    path_model + model_in_path)  ## каждый раз инициализируем модель в колабе иначе выдает ошибочный результат
                results, all_boxes, orig_shape = get_boxes(model.predict(video_source + f'{N}.mp4',
                                                                         line_thickness=2, vid_stride=1, save=True))
                np.save(path_model + f"{N}.npy", np.array((orig_shape, all_boxes)))
        except:
            print(f'Видео {N}: отсутствует')


def change_bbox(bbox, tail):  # функция для изменения размеров бокса подаваемого в треккер
    y2 = bbox[:, [1]] + tail

    bbox[:, [3]] = y2

    return bbox


def forward(bbox, tracks,
            fwd=False):  # эта функция позволяет сохранить лист детекций в который внесены айди от трека сохраняя нетрекованные боксы на случай последующей перетрековки
    person = np.empty((0, 8))  # Создадим пустой массив для каждого кадра который будем наполнять
    for i, bb in enumerate(bbox):  # Сравним каждый первичный не треккованный бокс
        for k, t in enumerate(tracks):  # С каждым треккованым
            if round(t[0]) == round(bb[0]) and round(t[1]) == round(bb[1]) and round(t[2]) == round(bb[2]) and round(
                    t[3]) == round(bb[3]):  # Если у них совпадают координаты
                bb_tr = np.copy(bb)
                bb_tr = np.insert(bb_tr, 6, t[
                    4])  # Добавляем к нетрекованному боксу трек определенный треккером (таким образом сохраняя конфиденс и класс)
                person = np.vstack((person,
                                    bb_tr))  # Складываем массив. На этом этапе остались в стеке только трекованные боксы. Но нам хотелось бы сохранить их все для фиксации нарушений или последующей перетрековки
            else:
                pass
        if fwd:
            if sum(np.in1d(bb[:4], tracks[:,
                                   :4])) < 4:  # добавим в оттрекованный массив то что треккер отсеял (на случай перетрековки)
                person = np.vstack((person, np.insert(bb, 6, -1)))

    return person


def tracking_on_detect(all_boxes, tracker,
                       orig_shp):  # эта функция отправляет нетрекованные боксы людей в треккер прокидывая остальные классы мимо треккера
    all_boxes_tr = np.empty((0, 8))
    for i in range(int(max(all_boxes[:, -1]))):
        bbox = all_boxes[all_boxes[:, -1] == i]
        bbox_unif = bbox[np.where(bbox[:, 5] != 0)][:,
                    :6]  # отбираем форму и каски в отдельный массив который прокинем мимо трека
        bbox_unif = np.hstack(
            (bbox_unif, np.tile(np.nan, (bbox_unif.shape[0], 1))))  # добавляем столбец с айди нан для касок и жилетов
        bbox_unif = np.hstack((bbox_unif, np.tile(i, (bbox_unif.shape[0], 1))))  # сохраняем номер кадра
        bbox = bbox[np.where(bbox[:, 5] == 0)]  # в трек идут только люди
        # bbox = change_bbox(bbox, tail)
        tracks = tracker.update(bbox[:, :-2], img_size=orig_shp, img_info=orig_shp)  # трекуем людей
        person = forward(bbox, tracks,
                         fwd=False)  # эта функция позволяет использовать далее лист детекций в который внесены айди от трека (трек фильтрует и удаляет боксы)
        all_boxes_tr = np.vstack((all_boxes_tr, person))  # складываем людей в массив
        all_boxes_tr = np.vstack((all_boxes_tr, bbox_unif))  # складываем каски и жилеты в массив
    return all_boxes_tr


def create_video_with_bbox(bboxes, video_source, video_out):  # функция отрисовки боксов на соответсвующем видео
    '''Функция записывает видео с рамками объектов, которые передаются в:
  bboxes - ndarray(x1, y1, x2, y2, conf, class, id, frame),
  если последовательность нарушена надо менять внутри функции.
  Другие обязательные аргументы функции:
  video_source - полный путь до исходного видео, на которое нужно наложить рамки;
  video_out - полный путь вновь создаваемого видео. Путь должен быть,
  а файла - не должно быть'''
    vid_src = cv2.VideoCapture(video_source)
    if vid_src.isOpened():
        # Разрешение кадра
        frame_size = (int(vid_src.get(3)), int(vid_src.get(4)))
        # Количество кадров в секунду
        fps = int(vid_src.get(5))
        # Количество кадров в файле
        len_frm = int(vid_src.get(7))
        # Выходное изображение записываем
        vid_out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'XVID'),
                                  fps, frame_size)
        # Пройдемся по всем кадрам
        for i in range(len_frm):
            ret, frame = vid_src.read()
            # На всякий пожарный случай выход
            if not ret: break
            # Отбираем рамки для кадра
            bbox = bboxes[bboxes[:, -1] == i, :-1]
            if len(bbox) > 0:
                # Только люди
                pbox = bbox[bbox[:, 5] == 0]
                for p in pbox:
                    # Добавим рамки
                    x1, y1 = int(p[0]), int(p[1])
                    x2, y2 = int(p[2]), int(p[3])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (153, 153, 153), 2)
                    # Добавим надпись в виде идентификатора объекта  и conf
                    msg = 'id' + str(int(p[6])) + ' ' + str(round(p[4], 2))
                    (w, h), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)
                    cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), (153, 153, 153), -1)
                    cv2.putText(frame, msg, (x1, y1 - 5), cv2.FONT_HERSHEY_DUPLEX, 0.45,
                                (255, 255, 255), 1)
                # Отрисовываем рамки касок
                helmets = bbox[bbox[:, 5] == 1]
                for helmet in helmets:
                    x1, y1 = int(helmet[0]), int(helmet[1])
                    x2, y2 = int(helmet[2]), int(helmet[3])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (51, 255, 204), 2)
                # Отрисовываем рамки жилетов
                vests = bbox[bbox[:, 5] == 2]
                for vest in vests:
                    x1, y1 = int(vest[0]), int(vest[1])
                    x2, y2 = int(vest[2]), int(vest[3])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (51, 204, 255), 2)
            # Записываем кадр
            vid_out.write(frame)
        # Освобождаем видео
        vid_out.release()
        vid_src.release()
    else:
        print('Видеофайл недоступен')
    # Заклинание cv2
    cv2.destroyAllWindows()


def get_men(out_boxes):
    men = np.empty((0, 9))
    inter_helm = 60  # поле вокруг бб человека для детекции касок
    inter_unif = 15  # поле вокруг бб человека для детекции жилетов
    for i in range(int(max(out_boxes[:, -1]))):

        # Только люди
        humans = out_boxes[(out_boxes[:, -3] == 0) & (out_boxes[:, -2] != -1) & (out_boxes[:, -1] == i)]
        # Только каски
        helmets = out_boxes[(out_boxes[:, -3] == 1) & (out_boxes[:, -1] == i)]
        # Только жилетки
        vests = out_boxes[(out_boxes[:, -3] == 2) & (out_boxes[:, -1] == i)]

        # Персональный подход к каждому человеку
        for man in humans:
            # Сколько касок в пределах рамок человека (либо 1, либо 0)
            helmet = 1 if len(helmets[(helmets[:, 0] >= man[0] - inter_helm) & (helmets[:, 1] >= man[1] - inter_helm) &
                                      (helmets[:, 2] <= man[2] + inter_helm) & (
                                              helmets[:, 3] <= man[3] + inter_helm)]) >= 1 else 0
            # Сколько жилеток в пределах рамок человека (либо 1, либо 0)
            vest = 1 if len(vests[(vests[:, 0] >= man[0] - inter_unif) & (vests[:, 1] >= man[1] - inter_unif) &
                                  (vests[:, 2] <= man[2] + inter_unif) & (
                                          vests[:, 3] <= man[3] + inter_unif)]) >= 1 else 0
            # Это просто добавление в массив men. Часть параметров нужны нам дважды для выявления макс и мин.
            # Поэтому дважды повторяются ордината низа и номер кадра
            men = np.vstack((men, np.array([man[-2],
                                            man[1], man[1],
                                            man[3], man[3],
                                            i, i,
                                            helmet, vest])))
            # Формируем датафрейм. Будем считать низ вначале и в конце, также первый кадр
            # и последний кадр, где был этот id (пока это одно и тоже значение)
    return men


def get_count_men(men, orig_shape):  # определяем все сами
    n_ = int(max(men[:, 0]))
    orig_shape = int(orig_shape)
    incoming = 0  # количество вошедших
    exiting = 0  # количество вышедших
    barier = 337

    gate_y = barier * orig_shape / 640  # определяем барьер сами

    box_y_top = [None] + [list() for _ in range(int(n_))]
    box_y_bottom = [None] + [list() for _ in range(int(n_))]
    box_frame = [None] + [list() for _ in range(int(n_))]

    for i, m in enumerate(men):
        id = int(m[0])
        frame_n = int(m[-4])
        box_frame[id].append(frame_n)

    if len(box_frame[id]) < 21:  # удаляем айди чей трек короче n кадров
        men = men[~np.isin(men[:, 0], id)]

    # иначе возникает исключение при поиске max
    if len(men[:, 0]) == 0:
        return men, incoming, exiting

    n = int(max(men[:, 0]))
    human_c = [None] + [list() for _ in range(int(n))]
    for m in men:
        num = int(m[0])
        box_center = (float(m[4]) - float(m[2])) / 2 + float(m[2])
        human_c[num].append(box_center)

    ind = []
    for i, h in enumerate(human_c):
        if h and h[0] < gate_y < h[-1]:
            incoming += 1
            ind.append(i)
        elif h and h[0] > gate_y > h[-1]:
            exiting += 1
            ind.append(i)
    men = men[np.isin(men[:, 0], ind)]

    return men, incoming, exiting


def get_count_vialotion(men, orig_shape):  # step height определяем сами
    orig_shape = int(orig_shape)
    df = pd.DataFrame(men, columns=['id',
                                    'first_top_y', 'last_top_y',
                                    'first_bottom_y', 'last_bottom_y',
                                    'first_frame', 'last_frame',
                                    'helmet', 'uniform'])
    # Зададим правила агрегирующих функций
    agg_func = {'first_top_y': 'first', 'last_top_y': 'last',
                'first_bottom_y': 'first', 'last_bottom_y': 'last',
                'first_frame': 'first', 'last_frame': 'last',
                'helmet': 'mean', 'uniform': 'mean'
                }
    # Группируем по id
    df1 = df.groupby('id').agg(agg_func)
    # Чтобы не выводилось предупреждение
    pd.options.mode.chained_assignment = None
    # Определяем пройденное расстояние
    df1.loc[:, 'distance'] = df1.last_bottom_y - df1.first_bottom_y
    df2 = df1.copy()
    # Считаем входящих (сверху вниз)
    incoming = df2.loc[df2.distance > 0].shape[0]
    # Считаем выходящих
    exiting = df2.loc[df2.distance < 0].shape[0]

    V_helm = 0.15
    V_unif = 0.5
    dictinex = {'incoming': incoming, 'exiting': exiting}
    df2.loc[df2.helmet < V_helm, 'helmet'] = 0
    df2.loc[df2.helmet >= V_helm, 'helmet'] = 1
    df2.loc[df2.uniform < V_unif, 'uniform'] = 0
    df2.loc[df2.uniform >= V_unif, 'uniform'] = 1

    clothing_helmet = []  # соберем данные по одежде в отдельный массив для последующей проверки на этапе проверки P и R
    clothing_unif = []
    for i, ds in enumerate(df2.values):
        clothing_helmet.append(int(ds[6]))
        clothing_unif.append(int(ds[7]))
    violations = df2.loc[((df2.helmet == 0) | (df2.uniform == 0)),
    ['helmet', 'uniform', 'first_frame', 'last_frame']]  # а это сами нарушения с номерами кадров

    return violations, incoming, exiting, df2, clothing_helmet, clothing_unif


"""
Эта функция позволяет пройтись по всем сохраненным ранее детекциям от start_vid до end_vid 
если они существуют в папке с моделью path_model оттрековать эти боксы, наложить на соответствующие
видео и записать результат постобработки в словари
"""


def track_on_detect(path_model, tracker_path, video_source, tracker, start_vid=1, end_vid=1):
    if end_vid == 1:
        length = len([f for f in os.listdir(path_model)
                      if f.endswith('.npy') and os.path.isfile(
                os.path.join(path_model, f))])  # подсчитаем количество видео в папке
    else:
        length = end_vid

    # создаем пустые словари которые будем наполнять предсказаниями вошедших /вышедших первым  и вторым алгоритмом
    d_in1 = {str(n): 0 for n in list(range(start_vid, length + 1))}
    d_out1 = {str(n): 0 for n in list(range(start_vid, length + 1))}
    d_in2 = {str(n): 0 for n in list(range(start_vid, length + 1))}
    d_out2 = {str(n): 0 for n in list(range(start_vid, length + 1))}

    # создаем пустые словари которые будем наполнять предсказаниями нарушениями по каске (используются в следующем разделе)
    # создаем пустые словари которые будем наполнять предсказаниями нарушениями по униформе (используются в следующем разделе)
    d_helmet = {str(n): [] for n in list(range(start_vid, length + 1))}
    d_unif = {str(n): [] for n in list(range(start_vid, length + 1))}

    for N in range(start_vid, length + 1):  # устанавливаем какие видео смотрим
        try:
            with open(path_model + f'{N}.npy',
                      'rb') as files:  # Загружаем объект содержащий формат исходного изображения и детекции
                all_boxes_and_shp = np.load(files, allow_pickle=True)
                orig_shp = all_boxes_and_shp[0]  # Здесь формат
                all_boxes = all_boxes_and_shp[1]  # Здесь боксы
                out_boxes = tracking_on_detect(all_boxes, tracker,
                                               orig_shp)  # Отправляем боксы в трекинг + пробрасываем мимо трекинга каски и нетрекованные боксы людей
                create_video_with_bbox(out_boxes, video_source + f'{N}.mp4',
                                       path_model + tracker_path + f'{N}_track.mp4')  # функция отрисовки боксов на соответсвующем видео
                # out_boxes_pd = pd.DataFrame(out_boxes)
                # out_boxes_pd.to_excel(path + tracker_path + f"df_{N}_{round(orig_shp[1])}_.xlsx") # сохраняем что бы было)
                men = get_men(
                    out_boxes)  # Смотрим у какого айди есть каски и жилеты (по порогу от доли кадров где был зафиксирован айди человека + каска и жилет в его бб и без них)
                men_clean, incoming1, exiting1 = get_count_men(men, orig_shp[
                    1])  # здесь переназначаем айди входящий/выходящий (временное решение для MVP, надо думать над продом)
                violation, incoming2, exiting2, df, clothing_helmet, clothing_unif = get_count_vialotion(men_clean,
                                                                                                         orig_shp[
                                                                                                             1])  # Здесь принимаем переназначенные айди смотрим нарушения а также повторно считаем входящих по дистанции, проверяем

                d_in1[f'{N}'] = incoming1
                d_out1[f'{N}'] = exiting1
                d_in2[f'{N}'] = incoming2
                d_out2[f'{N}'] = exiting2
                d_helmet[f'{N}'] = clothing_helmet
                d_unif[f'{N}'] = clothing_unif

        except:
            print(f'данные по видео {N}: отсутствуют')
    return d_in1, d_out1, d_in2, d_out2, d_helmet, d_unif
