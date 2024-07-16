import logging
from datetime import datetime
from typing import Optional

import numpy as np

from cluster import Clustering, gpu_availability
from encode import DuckDBClient
from optimize import (
    HyperparameterSearcher,
    Grid,
    CategoricalParameter,
    CategoricalSearchSpace,
    LogUniformParameter,
    NumericalSearchSpace,
    UniformParameter,
)


class MusicMetadataParamSearcher(HyperparameterSearcher):
    GRID = Grid(
        [
            # HDBSCAN parameters
            CategoricalParameter(
                name="cluster_selection_method",
                search_space=CategoricalSearchSpace(values=["eom", "leaf"]),
            ),
            LogUniformParameter(
                name="cluster_selection_epsilon",
                search_space=NumericalSearchSpace(low=0.1, high=2.0, step=0.1),
            ),
            UniformParameter(
                name="min_cluster_size",
                search_space=NumericalSearchSpace(low=5, high=300, step=5),
            ),
            UniformParameter(
                name="min_samples",
                search_space=NumericalSearchSpace(low=5, high=100, step=5),
            ),
            # UMAP Parameters
            UniformParameter(
                name="n_neighbors",
                search_space=NumericalSearchSpace(low=20, high=200, step=25),
            ),
            UniformParameter(
                name="n_components",
                search_space=NumericalSearchSpace(low=5, high=10, step=1),
            ),
            UniformParameter(
                name="min_dist",
                search_space=NumericalSearchSpace(low=0.0, high=1.0, step=0.01),
            ),
        ]
    )


class Experiment:
    def __init__(
        self,
        file_path: Optional[str] = None,
        n_samples: Optional[int] = None,
        seed: int = 42,
        alpha: float = 0.1,
        logger: logging.Logger = logging.getLogger(__name__),
        read_only: bool = False,
    ):
        np.random.seed(seed)

        self.n_samples = n_samples
        self.logger = logger
        self.seed = seed
        self.alpha = alpha

        self.param_searcher = MusicMetadataParamSearcher(
            file_path or f"history/loss/{datetime.now().strftime('%Y%m%d')}.json",
            logger=self.logger,
            metadata={
                "date": datetime.now().strftime("%Y-%m-%d"),
                "n_samples": n_samples,
                "alpha": alpha,
                "seed": seed,
            },
        )

        self.n_samples = n_samples
        self.client = DuckDBClient(read_only=read_only)

    def run(self, n_evaluations: int = 10):
        embeddings = self.client.get_embeddings(self.n_samples, self.seed)
        self.logger.info(f"Using {'GPU' if gpu_availability() else 'CPU'}.")

        def sample_to_loss(sample):
            model = Clustering(
                params=sample, logger=self.logger, seed=self.seed, alpha=self.alpha
            )
            model.fit(embeddings)
            return model.get_loss(embeddings)

        self.param_searcher.search(sample_to_loss, n_evaluations=n_evaluations)

    def set_clusters(self):
        """
        Assign clusters using fine-tuned clustering model.
        """
        tracks = self.client.get_tracks(self.n_samples, self.seed)
        params = self.param_searcher.get_best_observation()
        self.logger.info(f"Best observation: {params}.")

        model = Clustering(params=params["sample"], seed=self.seed, alpha=self.alpha)

        embeddings = self.client.get_embeddings(self.n_samples, self.seed)
        model.fit(embeddings)

        recorded_loss = params["loss"]
        fitted_loss = model.get_loss(embeddings)

        # Ensure we measure same loss
        self.logger.info(f"Fitted loss: {fitted_loss}.")
        if not np.isclose(fitted_loss, recorded_loss):
            self.logger.warning(
                f"Losses fitted and retrieved from prior experiment differ "
                f"(Recorded loss: {recorded_loss}, fitted loss: {fitted_loss})."
            )

        tracks["cluster"] = model.labels

        self.client.insert_clusters(tracks["cluster"].reset_index())


if __name__ == "__main__":
    e = Experiment(n_samples=1_000, read_only=True)
    e.run(n_evaluations=5)
    # e.set_clusters()
