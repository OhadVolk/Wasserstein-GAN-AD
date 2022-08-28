from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    def __init__(self, data: pd.DataFrame, y: pd.Series):
        self.data = data
        self.labels = y

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        data = torch.Tensor(self.data.iloc[idx])
        labels = self.labels.iloc[idx]

        return data, labels

    def __len__(self):
        return len(self.data)
