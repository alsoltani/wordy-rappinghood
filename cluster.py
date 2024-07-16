import logging
from dataclasses import dataclass
from enum import Enum
from importlib.util import find_spec
from typing import Optional, Tuple, Union, Dict

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from hdbscan import HDBSCAN
from hdbscan.prediction import all_points_membership_vectors
from marshmallow_dataclass import class_schema
from sklearn.manifold import trustworthiness
from umap import UMAP

from utils import custom_dbcv

if find_spec("cuml"):
    from cuml.cluster import HDBSCAN as cuHDBSCAN  # pylint: disable=E0611, E0401
    from cuml.manifold.umap import UMAP as cuUMAP  # pylint: disable=E0611, E0401
    from cuml.cluster.hdbscan.prediction import all_points_membership_vectors

logging.basicConfig(level=logging.INFO)


class SelectionMethod(Enum):
    EOM = "eom"
    LEAF = "leaf"


class CategoricalDistance(Enum):
    JACCARD = "jaccard"
    HAMMING = "hamming"


class NumericalDistance(Enum):
    MANHATTAN = "manhattan"
    EUCLIDEAN = "euclidean"
    COSINE = "cosine"


class ModelComponent(Enum):
    UMAP_VECTOR = "UMAP_VECTOR"
    UMAP_NUMERICS = "UMAP_NUMERICS"
    UMAP_COMBINATION = "UMAP_COMBINATION"
    HDBSCAN = "HDBSCAN"


class ClusteringEvaluation(Enum):
    RELATIVE_VALIDITY = "relative_validity"


@dataclass
class HDBSCANParams:
    cluster_selection_method: SelectionMethod
    cluster_selection_epsilon: float
    min_cluster_size: int
    min_samples: int
    gen_min_span_tree: bool = True


@dataclass
class UMAPParams:
    n_neighbors: int
    n_components: int
    min_dist: float


hdbscan_schema = class_schema(HDBSCANParams)()
umap_schema = class_schema(UMAPParams)()


def gpu_availability():
    return find_spec("cuml") and torch.cuda.is_available()


class Clustering:
    """
    UMAP on textual embeddings + HDBSCAN clustering.
    """

    def __init__(
        self,
        params: Dict[str, Union[float, str]],
        seed: int = 42,
        alpha: float = 0.0,
        logger: logging.Logger = logging.getLogger(__name__),
    ):
        np.random.seed(seed)

        self.seed = seed
        self.alpha = alpha
        self.logger = logger
        self.use_gpu = gpu_availability()

        self.evaluation_method: ClusteringEvaluation = (
            ClusteringEvaluation.RELATIVE_VALIDITY
        )
        self.umap_params = None
        self.hdbscan_params = None

        self.umap = None
        self.hdbscan = None
        self.embedding = None

        self.labels = None
        self.probabilities = None

        self.init_models(params)

    def init_models(self, params: Dict[str, Union[float, str]]):
        # Query params from Bender experiment
        hdbscan_params, umap_params = self.get_params(**params)
        self.hdbscan_params = self.process_params(hdbscan_schema.dump(hdbscan_params))
        self.umap_params = self.process_params(umap_schema.dump(umap_params))

        # Init models
        if self.use_gpu:
            self.umap = cuUMAP(
                random_state=self.seed,
                verbose=False,
                metric="cosine",
                # FIXME: 'spectral' initialization, using a spectral embedding of the fuzzy 1-skeleton,
                #  does not produce deterministic results when random_state is specified.
                init="random",
                **self.umap_params,
            )
            self.hdbscan = cuHDBSCAN(
                # Optimization to speed up the prediction queries later.
                # This is only useful when planning to predict clusters for new points.
                prediction_data=True,
                **self.hdbscan_params,
            )

        else:
            self.umap = UMAP(
                random_state=self.seed,
                n_jobs=1 if self.seed is not None else -1,
                metric="cosine",
                verbose=False,
                low_memory=True,
                init="random",
                **self.umap_params,
            )
            self.hdbscan = HDBSCAN(
                prediction_data=True,
                **self.hdbscan_params,
            )

    def fit(self, vector_embeddings: pd.DataFrame):
        self.umap.fit(vector_embeddings)
        self.embedding = self.umap.embedding_

        self.hdbscan.fit(self.embedding)
        self.labels = self.hdbscan.labels_
        self.probabilities = self.soft_clustering_probabilities()

    def soft_clustering_probabilities(self) -> npt.NDArray:
        assert self.labels is not None
        if len(np.unique(self.labels)) == 1:
            un_normalized = np.ones_like(self.labels, dtype="float")
        else:
            un_normalized = all_points_membership_vectors(self.hdbscan)
        return self.ensure_matrix_and_normalize(un_normalized)

    @staticmethod
    def ensure_matrix_and_normalize(a: npt.NDArray) -> npt.NDArray:
        arr = np.atleast_2d(a)
        if arr.shape[0] == 1:
            arr = arr.T

        return arr / np.sum(arr)
        # return (arr.T / np.sum(arr, axis=1)).T

    def get_loss(self, data: pd.DataFrame, n_neighbors: Optional[int] = None) -> float:
        """
        Clustering loss + Shannon entropy regularization term
        over soft clustering probabilities.
        """
        return self.clustering_loss(data, n_neighbors) + self.alpha * self.entropy_loss(
            self.probabilities
        )

    @staticmethod
    def shannon_entropy(a: npt.NDArray) -> float:
        """
        Average normalized Shannon entropy per cluster
        :param a: array input (n_samples, n_clusters).
        :return: entropy, in [0, 1].
        """
        n_samples = a.shape[0]

        if n_samples > 1:
            return -np.sum(
                a * np.log2(a, out=np.zeros_like(a), where=(a > 0)),
                axis=0,
            ).mean() / np.log2(n_samples)

        return 1

    def entropy_loss(self, a: npt.NDArray) -> float:
        return 1 - self.shannon_entropy(a)

    def clustering_loss(
        self, data: pd.DataFrame, n_neighbors: Optional[int] = None
    ) -> float:
        """
        Loss based on manifold trustworthiness + density-based clustering validity.
        :param data: input data
        :param n_neighbors: included here if you want to compare
        loss with same n_neighbors on different hyperparameter sets.
        Defaults to value suggested by benderopt.
        :return:
        """
        if self.evaluation_method == ClusteringEvaluation.RELATIVE_VALIDITY:
            if isinstance(self.labels, np.ndarray):
                labels = self.labels
            else:
                labels = self.labels.values

            # Metric "cosine" is currently unsupported by cuML,
            # so we'll revert back to sklearn
            return (
                1
                - trustworthiness(
                    data,
                    self.embedding,
                    n_neighbors=n_neighbors or self.umap_params["n_neighbors"],
                    metric="cosine",
                )
                * (custom_dbcv(self.hdbscan.minimum_spanning_tree_, labels) + 1)
                / 2
            )

        else:
            raise NotImplementedError(
                f"Clustering loss for method {self.evaluation_method} has not been implemented."
            )

    @staticmethod
    def get_params(
        cluster_selection_method: str,
        cluster_selection_epsilon: float,
        min_cluster_size: int,
        min_samples: int,
        n_neighbors: int,
        n_components: int,
        min_dist: float,
    ) -> Tuple[HDBSCANParams, UMAPParams]:
        return (
            HDBSCANParams(
                cluster_selection_method=SelectionMethod(cluster_selection_method),
                cluster_selection_epsilon=float(cluster_selection_epsilon),
                min_cluster_size=int(min_cluster_size),
                min_samples=int(min_samples),
            ),
            # UMAP on text embeddings
            UMAPParams(
                n_neighbors=int(n_neighbors),
                n_components=int(n_components),
                min_dist=min_dist,
            ),
        )

    def process_params(self, obj: Union[Dict, str, float]) -> Union[Dict, str, float]:
        if isinstance(obj, dict):
            return {key: self.process_params(value) for key, value in obj.items()}
        elif isinstance(obj, str):
            return obj.lower()
        else:
            return obj
