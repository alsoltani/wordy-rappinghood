import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer


def assert_numerical_format(data: pd.DataFrame):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input should be a pandas DataFrame.")
    if data.empty:
        raise ValueError("Input DataFrame should not be empty.")

    numerical = data.select_dtypes(include=["float", "int"])
    if numerical.shape[1] == 0:
        raise ValueError("No numerics found. Check numerics are in dataframe.")


def preprocess_numerical(
    data: pd.DataFrame, impute_strategy: str = "median", **kwargs
) -> pd.DataFrame:
    assert_numerical_format(data)
    numerical = impute_numerical(data, strategy=impute_strategy, **kwargs)
    return transform_numerical(numerical)


def impute_numerical(
    numerical: pd.DataFrame, strategy: str = "median", **kwargs
) -> pd.DataFrame:
    """
    Imputes numerical features with given strategy.
    """

    if strategy not in ["median", "mean"]:
        raise NotImplementedError(f"Invalid strategy for numerical: {strategy}.")

    if numerical.isnull().sum().sum() == 0:
        return numerical

    imputer = SimpleImputer(strategy=strategy, **kwargs)
    numerical_imputed = pd.DataFrame(
        imputer.fit_transform(numerical), columns=numerical.columns
    )

    return numerical_imputed


def transform_numerical(numerical: pd.DataFrame) -> pd.DataFrame:
    """
    Transform numerical data to make it more Gaussian-like.
    """
    for name in numerical.columns.tolist():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            pt = PowerTransformer(copy=False)
            numerical[name] = pt.fit_transform(np.array(numerical[name]).reshape(-1, 1))

    return numerical
