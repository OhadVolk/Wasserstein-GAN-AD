from typing import Tuple

import numpy as np
import pandas as pd
from pyod.models.base import BaseDetector


def train(model: BaseDetector, X: pd.DataFrame, y: pd.Series) -> BaseDetector:
    model.fit(X, y)
    return model


def predict(model: BaseDetector, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return y_pred, y_proba
