import os
from functools import partial
from typing import List
from constants import ROOT, TARGET_COL
import pandas as pd

from data.utils import get_listdir_full_path, get_file_name_from_path, shuffle_df


def one_hot_encode(data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    return pd.get_dummies(data[columns])


def get_categorical_columns(data: pd.DataFrame) -> List[str]:
    return list(data.select_dtypes(exclude=['number']).columns)


def preprocess_dataset(dataset_path: str, target_column: str, preprocessed_path: str) -> None:
    dataset = pd.read_csv(dataset_path)
    dataset_name = get_file_name_from_path(dataset_path)
    print(f'Preprocessing: {dataset_name}')
    X = dataset.drop(target_column, axis=1)
    y = dataset[target_column]
    categorical_columns = get_categorical_columns(X)
    if categorical_columns:
        print(f'One hot encoding {categorical_columns}')
        one_hot_encoded_categorical = one_hot_encode(X, categorical_columns)
        X = pd.concat([X.drop(categorical_columns, axis='columns'), one_hot_encoded_categorical], axis=1)
    else:
        print('No categorical columns')
    preprocessed_dataset = shuffle_df(pd.concat([X, y], axis=1))
    preprocessed_dataset.to_csv(os.path.join(preprocessed_path, dataset_name), index=False)


def preprocess_datasets(datasets_path: List[str], target_column: str, preprocessed_path: str) -> None:
    partial_preprocess_dataset = partial(preprocess_dataset, target_column=target_column,
                                         preprocessed_path=preprocessed_path)
    list(map(partial_preprocess_dataset, datasets_path))


def main():
    path = os.path.join(ROOT, 'Raw')
    datasets_path = get_listdir_full_path(path)
    preprocessed_path = os.path.join(ROOT, 'Preprocessed')
    preprocess_datasets(datasets_path, target_column=TARGET_COL, preprocessed_path=preprocessed_path)


if __name__ == '__main__':
    main()
