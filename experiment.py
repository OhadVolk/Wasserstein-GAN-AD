from time import time
from typing import Dict, List

import pandas as pd
from pyod.models.base import BaseDetector
from torch import nn, Module
from torch.utils.data import DataLoader
from constants import TARGET_COL
from dataset import TabularDataset
from metrics import get_metrics
from model_engine import predict
from split import split_data_normal_train_mixed_val
from utils import get_duration
from wgan import Generator, Discriminator


def experiment(dataset_path: str, train_frac: float, models: Dict[str, BaseDetector], random_state: int) -> List[
    Dict[str, float]]:
    data = pd.read_csv(dataset_path)
    train, val = split_data_normal_train_mixed_val(data, train_frac, random_state=random_state)

    X_train, y_train = train.drop(columns=[TARGET_COL]), train[TARGET_COL]
    X_val, y_val = val.drop(columns=[TARGET_COL]), val[TARGET_COL]

    results_list = []
    for model_name, model in models.items():
        t0 = time()

        model = model.fit(X_train)
        val_preds, val_probas = predict(model, X_val)
        metrics = get_metrics(y_val, val_preds, val_probas)

        t1 = time()
        duration = get_duration(t0, t1)

        metrics['duration'] = duration
        metrics['model_name'] = model_name
        results_list.append(metrics)
    return results_list


