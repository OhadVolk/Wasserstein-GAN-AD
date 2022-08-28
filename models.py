from typing import Dict

from pyod.models.base import BaseDetector
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.inne import INNE
from pyod.models.knn import KNN
from pyod.models.ecod import ECOD
from pyod.models.loda import LODA
from pyod.models.lunar import LUNAR
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.pca import PCA
from pyod.models.gmm import GMM


def get_models() -> Dict[str, BaseDetector]:
    return {'Isolation Forest': IForest(),
            'LOF': LOF(),
            'HBOS': HBOS(),
            'INNE': INNE(),
            'KNN': KNN(),
            'ECOD': ECOD(),
            'LODA': LODA(),
            'LUNAR': LUNAR(),
            'DeepSVDD': DeepSVDD(),
            'PCA': PCA(),
            'GMM': GMM()}
