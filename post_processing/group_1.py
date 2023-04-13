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


if __name__ == '__main__':
    demo()
