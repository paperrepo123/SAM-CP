import numpy as np
# import torch


def x1y1wh_to_xyxy(bbox):
    return np.stack([bbox[:, 0], bbox[:, 1], bbox[:, 2] + bbox[:, 0], bbox[:, 3] + bbox[:, 1]],axis=1)
def xyxy_to_x1y1wh(bbox):
    return np.stack([bbox[:, 0], bbox[:, 1], bbox[:, 2] - bbox[:, 0], bbox[:, 3] - bbox[:, 1]],axis=1)
