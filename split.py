import pandas as pd

from constants import TARGET_COL


def separate_normal_and_anomalous(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    normal = df[df[TARGET_COL] == 0]
    anomalous = df[df[TARGET_COL] == 1]
    return normal, anomalous


def split_data_normal_train_mixed_val(df: pd.DataFrame, train_frac: float, random_state: int) -> (
        pd.DataFrame, pd.DataFrame, pd.DataFrame):
    normal, anomalous = separate_normal_and_anomalous(df)
    train = normal.sample(frac=train_frac, random_state=random_state)
    val_normal = normal.drop(train.index)
    val = pd.concat([val_normal, anomalous], axis=0)
    return train, val
