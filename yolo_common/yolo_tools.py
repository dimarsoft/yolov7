import numpy as np
import torch


def xyxy2ltwh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [left, top, w, h]
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0]  # left
    y[:, 1] = x[:, 1]  # top
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def ltwh2xyxy(x):
    # Convert nx4 boxes from [left, top, w, h] to [x1, y1, x2, y2]
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0]  # left x
    y[:, 1] = x[:, 1]  # top  y
    y[:, 2] = x[:, 0] + x[:, 2]  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3]  # bottom right y
    return y
