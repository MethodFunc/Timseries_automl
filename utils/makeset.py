import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

"""
Time series data make set 
auth: Methodfunc - Kwak Piljong
date: 2021.08.26
modify date: 2021.08.30
version: 0.2
describe: 슬라이딩 윈도우 합침. 기존 윈도우메이커 추가
"""


def split_data(data, train_size: float, val_size: float, test_size: float):
    if train_size + val_size + test_size > 1:
        raise "Each sum must not exceed 1."

    total_size = len(data)
    train = int(total_size * train_size)
    val = int(total_size * val_size)
    test = int(total_size * test_size)

    if train + val + test != total_size:
        other = total_size - (train + val + test)
        train = train + other

    train_data = data[:train]
    val_data = data[train : train + val]
    test_data = data[train + val :]

    assert total_size == train + val + test

    return train_data, val_data, test_data


def basicwindowmaker(features, targets, start_index, end_index, window_size):
    if isinstance(features, pd.core.frame.DataFrame) or isinstance(
        features, pd.core.series.Series
    ):
        features = features.values

    if isinstance(targets, pd.core.frame.DataFrame) or isinstance(
        targets, pd.core.series.Series
    ):
        targets = targets.values

    data, label = [], []

    start_index = start_index + window_size

    if end_index is None:
        end_index = len(features)

    for i in range(start_index, end_index):
        indices = range(i - window_size, i)
        data.append(features[indices])

        label.append(targets[i])

    data = np.array(data)
    label = np.array(label)

    return data, label


class DataMaker(Dataset):
    def __init__(self, x, y, window_size=None, sliding=True):
        super(DataMaker, self).__init__()
        self.x_data = x
        self.y_data = y
        self.window_size = window_size
        self.sliding = sliding

        if self.sliding and (self.window_size is None):
            raise "sliding is True. need to window size"

        if self.sliding:
            self.sliding_window()

    def __len__(self):
        return len(self.x_data)

    def __repr__(self):
        return f"Input x_data types: {type(self.x_data)}\nInput y_data types: {type(self.y_data)}"

    def __getitem__(self, idx):
        x_data = torch.FloatTensor(self.x_data[idx])
        y_data = torch.FloatTensor(self.y_data[idx])

        return x_data, y_data

    def sliding_window(self):
        if isinstance(self.x_data, pd.core.frame.DataFrame) or isinstance(
            self.x_data, pd.core.series.Series
        ):
            self.x_data = self.x_data.values

        if isinstance(self.y_data, pd.core.frame.DataFrame) or isinstance(
            self.y_data, pd.core.series.Series
        ):
            self.y_data = self.y_data.values

        self.x_data = self.x_data[
            np.arange(self.window_size)[None, :]
            + np.arange(self.x_data.shape[0] - self.window_size)[:, None]
        ]
        self.y_data = self.y_data[self.window_size :]
