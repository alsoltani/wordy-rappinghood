import dataclasses
import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Union

import numpy as np
import pandas as pd
from benderopt.base import OptimizationProblem, Observation
from benderopt.optimizer import optimizers


class ParameterCategory(Enum):
    CATEGORICAL = "categorical"
    UNIFORM = "uniform"
    LOGUNIFORM = "loguniform"


@dataclass
class Parameter:
    name: str


@dataclass
class CategoricalSearchSpace:
    values: List[str]


@dataclass
class NumericalSearchSpace:
    low: float
    high: float
    step: float = 1


@dataclass
class CategoricalParameter(Parameter):
    search_space: CategoricalSearchSpace
    category: ParameterCategory = ParameterCategory.CATEGORICAL.value


@dataclass
class UniformParameter(Parameter):
    search_space: NumericalSearchSpace
    category: ParameterCategory = ParameterCategory.UNIFORM.value


@dataclass
class LogUniformParameter(Parameter):
    search_space: NumericalSearchSpace
    category: ParameterCategory = ParameterCategory.LOGUNIFORM.value


@dataclass
class Grid:
    parameters: List[Parameter]


class HyperparameterSearcher:
    """
    Parzen Tree Optimizer for hyperparameter selection.
    """

    GRID = None

    def __init__(
        self,
        file_name: str,
        logger: logging.Logger = logging.getLogger(__name__),
        metadata: Optional[Dict[str, Union[str, int, float]]] = None,
    ):
        self.history_file = Path(file_name)
        self.optimization_problem = None
        self.init_optimization_problem(metadata)

        self.optimizer = optimizers["parzen_estimator"](self.optimization_problem)
        self.logger = logger

    def search(self, sample_to_loss_function, n_evaluations: int = 10) -> None:
        """
        :param sample_to_loss_function:
        function to return loss based on selected sample, i.e. as a partial
        f: hyperparameter_sample -> loss(data, hyperparameter_sample),
        where data is set.
        :param n_evaluations: number of evaluations to perform search on
        :return:
        """

        with open(self.history_file, "r") as file:
            history = json.load(file)

        training_times: List[float] = []
        history.setdefault("observations", [])

        for _ in range(n_evaluations):
            sample = self.optimizer.suggest()

            start_time = time.time()
            loss = sample_to_loss_function(sample)
            end_time = time.time()

            training_times.append(end_time - start_time)
            obs_dict = {"loss": loss, "sample": sample}
            self.logger.info(obs_dict)

            observation = Observation.from_dict(obs_dict)
            self.optimization_problem.add_observation(observation)
            history["observations"].append(obs_dict)

            with open(self.history_file, "w") as file:
                json.dump(history, file, indent=4)

        self.logger.info(
            f"Average training time ({n_evaluations} evaluations): {np.mean(training_times):.2f} seconds."
        )

    def init_optimization_problem(
        self, metadata: Optional[Dict[str, Union[str, int, float]]] = None
    ):
        if self.GRID is None:
            raise ValueError(
                "Please provide a grid to perform hyperparameter selection."
            )

        if not self.history_file.exists():
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            self.history_file.touch()

        with open(self.history_file, "r") as file:
            history = json.load(file)

        with open(self.history_file, "w") as file:
            json.dump(
                {
                    "parameters": dataclasses.asdict(self.GRID)["parameters"],
                    "observations": history.get("observations", []),
                    "metadata": metadata or {},
                },
                file,
                indent=4,
            )

        self.optimization_problem = OptimizationProblem.from_json(self.history_file)

    def get_best_observation(self, index: int = 0):
        with open(self.history_file, "r") as res_file:
            history = json.load(res_file)
            obs_list = [obs for obs in history["observations"] if pd.notna(obs["loss"])]
            if not obs_list:
                raise ValueError(f"No observations found for file {self.history_file}.")
            return sorted(obs_list, key=lambda x: x["loss"])[index]

    def get_best_sample(self, index: int = 0):
        return self.get_best_observation(index)["sample"]

    def get_best_loss(self, index: int = 0):
        return self.get_best_observation(index)["loss"]
