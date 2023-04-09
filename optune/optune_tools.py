import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from optuna.trial import FrozenTrial


def reset_seed(seed=123):
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def save_result(trial: FrozenTrial, save_folder: str, tag: str, use_date: bool = True) -> None:
    """
    Сохраняет результаты работы Optune
    Args:
        use_date(bool): Дополнить имя файла датой
        tag: Тип трекера, который использовали. Будет добавлено к имени файла
        trial(FrozenTrial): результат работы Optune
        save_folder: Папка, в которой будут сохранены результаты: value, param

    Returns:

    """

    if use_date:
        now = datetime.now()
        tag = f"{now.year:04d}_{now.month:02d}_{now.day:02d}_{now.hour:02d}_{now.minute:02d}_" \
              f"{now.second:02d}_{tag}"

    value_json_file = Path(save_folder) / f"{tag}_value.json"

    params_json_file = Path(save_folder) / f"{tag}_params.json"

    with open(value_json_file, "w") as write_file:
        write_file.write(json.dumps(trial.value, indent=4, sort_keys=True))

    with open(params_json_file, "w") as write_file:
        write_file.write(json.dumps(trial.params, indent=4, sort_keys=True))
