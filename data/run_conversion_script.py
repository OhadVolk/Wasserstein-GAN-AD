import os

import scipy.io
import pandas as pd
from constants import ROOT
from data.utils import get_listdir_full_path


def convert_mat_to_pd(path: str) -> None:
    print(f'Converting: {path}')
    data = scipy.io.loadmat(path)
    X = data['X']
    y = data['y']
    new_path = path.replace('.mat', '.csv')
    pd.concat([pd.DataFrame(X), pd.DataFrame(y, columns=['y'])], axis=1).to_csv(new_path, index=False)


def main():
    datatype = '.mat'
    datasets_paths = get_listdir_full_path(ROOT)
    matlab_datasets_paths = [path for path in datasets_paths if datatype in path]
    map(convert_mat_to_pd, matlab_datasets_paths)


if __name__ == '__main__':
    main()
