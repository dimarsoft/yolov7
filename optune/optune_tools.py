import json
import os
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import optuna
import torch
from optuna import Study
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial


def reset_seed(seed=123):
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def add_date_prefix(tag) -> str:
    now = datetime.now()
    tag = f"{now.year:04d}_{now.month:02d}_{now.day:02d}_{now.hour:02d}_{now.minute:02d}_" \
          f"{now.second:02d}_{tag}"
    return tag


def save_result(trial: FrozenTrial, save_folder: str, tag: str, use_date: bool = True, study: Study = None) -> None:
    """
    Сохраняет результаты работы Optune
    Args:
        study(Study):
        use_date(bool): Дополнить имя файла датой
        tag: Тип трекера, который использовали. Будет добавлено к имени файла
        trial(FrozenTrial): результат работы Optune
        save_folder: Папка, в которой будут сохранены результаты: value, param

    Returns:

    """

    if use_date:
        tag = add_date_prefix(tag)

    optune_json_file = Path(save_folder) / f"{tag}_optune.json"

    common_dic = {
        "Value": trial.value,
        "params": trial.params,
        "number": trial.number,
    }

    if study is not None:
        common_dic["study.direction"] = str(study.direction)
        common_dic["study.best_trial.params"] = study.best_trial.params
        common_dic["study.best_trial.params.number"] = study.best_trial.number
        common_dic["study.best_value"] = study.best_value
        common_dic["study.user_attrs"] = study.user_attrs

    with open(optune_json_file, "w") as write_file:
        write_file.write(json.dumps(common_dic, indent=4, sort_keys=True))


def save_callback(study: Study, trial: FrozenTrial) -> None:
    """

    callback для сохранения информации, вызывается каждый триал в оптуне.
    До запуска обязательно нужно указать аттрибуты:

    study.set_user_attr("save_folder", output_folder)
    study.set_user_attr("tag", tag)


    Args:
        study:
        trial:

    Returns:

    """
    save_folder = study.user_attrs["save_folder"]
    tag = study.user_attrs["tag"]

    save_result(trial, save_folder, tag, use_date=False, study=study)


def common_run_optuna(tracker_tag: str, output_folder: str, objective: Callable, trials=100) -> Study:
    """

    Args:
        tracker_tag: Название трекера, используется при создании файла с результатами
        output_folder: Папка для сохранения результата
        objective: Функция перебора параметров
        trials: Количество итераций n_trials

    Returns:
        Study

    """
    study = optuna.create_study(direction=StudyDirection.MAXIMIZE)

    tag = add_date_prefix(tracker_tag)

    study.set_user_attr("save_folder", output_folder)
    study.set_user_attr("tag", tag)

    study.optimize(objective, n_trials=trials, show_progress_bar=True, callbacks=[save_callback])

    trial = study.best_trial

    print(f"Accuracy: {trial.value}")
    print(f"Best hyper parameters: {trial.params}")

    save_result(trial, output_folder, tag, use_date=False, study=study)

    return study
