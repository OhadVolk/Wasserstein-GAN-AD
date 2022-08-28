import os.path
from functools import partial
from typing import List, Dict

import pandas as pd

from data.utils import get_file_name_from_path
from experiment import experiment
from models import get_models


def load_data(path: str) -> pd.DataFrame:
    if path.endswith('.csv'):
        return pd.read_csv(path)
    elif path.endswith('.xlsx'):
        return pd.read_excel(path)
    elif path.endswith('.parquet'):
        return pd.read_parquet(path)


def run_dataset_pipeline(dataset_path: str, train_frac: float, random_state: int) -> pd.DataFrame:
    models = get_models()
    results_list = experiment(dataset_path, train_frac, models, random_state=random_state)
    return pd.DataFrame(results_list)


def run_dataset_multiple_pipeline(dataset_path: str, train_frac: float, random_state: int,
                                  num_iterations: int) -> pd.DataFrame:
    dataset_multiple_result_list = [run_dataset_pipeline(dataset_path, train_frac, random_state + i) for i in
                                    range(num_iterations)]
    return pd.concat(dataset_multiple_result_list)


def run_datasets_multiple_pipeline(datasets_path: List[str], train_frac: float, random_state: int,
                                   num_iterations: int) -> Dict[str, pd.DataFrame]:
    partial_dataset_multiple_pipeline = partial(run_dataset_multiple_pipeline, train_frac=train_frac,
                                                random_state=random_state, num_iterations=num_iterations)
    dataset_multiple_result_dict = {
        get_file_name_from_path(dataset_path): partial_dataset_multiple_pipeline(dataset_path) for dataset_path in
        datasets_path}
    return dataset_multiple_result_dict


def run_pipeline(datasets_path: List[str], traic_frac: float, random_state: int, num_iterations: int,
                 results_path: str) -> None:
    dataset_multiple_result_dict = run_datasets_multiple_pipeline(datasets_path, train_frac=traic_frac,
                                                                  random_state=random_state,
                                                                  num_iterations=num_iterations)
    for dataset_name, dataset_result in dataset_multiple_result_dict.items():
        dataset_result.to_csv(os.path.join(results_path, dataset_name + '.csv'))


def main():
    datasets_path = r"C:\Users\ovolk\Machine Learning\Datasets\Anomaly Detection\Preprocessed"
    # datasets_path_list = get_listdir_full_path(datasets_path)[:10 ]
    datasets_path_list = [r"C:\Users\ovolk\Machine Learning\Datasets\Anomaly Detection\Preprocessed\http.csv"]
    train_frac = 0.5
    random_state = 1337
    num_iterations = 10
    results_path = r"C:\Users\ovolk\Machine Learning\Datasets\Anomaly Detection\Results"
    run_pipeline(datasets_path_list, train_frac, random_state, num_iterations, results_path)




