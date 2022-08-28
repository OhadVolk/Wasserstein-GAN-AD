import os
from typing import List
from pathlib import Path
import pandas as pd


def get_listdir_full_path(path: str) -> List[str]:
    return [os.path.join(path, name) for name in os.listdir(path)]


def shuffle_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.sample(frac=1).reset_index(drop=True)


def get_file_name_from_path(path: str) -> str:
    return Path(path).stem
