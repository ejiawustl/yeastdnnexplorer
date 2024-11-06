import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LassoCV

# Import the functions from the module being tested
from yeastdnnexplorer.ml_models.lasso_modeling import (
    bootstrap_stratified_cv_modeling,
    examine_bootstrap_coefficients,
    generate_modeling_data,
    stratification_classification,
    stratified_cv_modeling,
)


# Sample Data for Tests
@pytest.fixture
def sample_data():
    response_df = pd.DataFrame(
        {"tf1": np.random.randn(100)}, index=[f"gene{i}" for i in range(100)]
    )
    predictors_df = pd.DataFrame(
        {f"tf{i}": np.random.randn(100) for i in range(1, 4)},
        index=[f"gene{i}" for i in range(100)],
    )
    return response_df, predictors_df


@pytest.fixture
def lasso_output(sample_data):
    response_df, predictors_df = sample_data
    y, X = generate_modeling_data("tf1", response_df, predictors_df)
    return bootstrap_stratified_cv_modeling(
        n_bootstraps=10, y=y, X=X, estimator=LassoCV()
    )


# Tests for generate_modeling_data
def test_generate_modeling_data_all(sample_data):
    response_df, predictors_df = sample_data
    y, X = generate_modeling_data("tf1", response_df, predictors_df)
    assert isinstance(y, pd.DataFrame)
    assert isinstance(X, pd.DataFrame)
    assert y.shape[0] == X.shape[0]
    assert "tf1" in y.columns
    assert all(
        [col in X.columns for col in ["tf1", "tf1:tf2", "tf1:tf3"]]
    ), "The predictor columns are not as expected."


# Tests for generate_modeling_data
def test_generate_modeling_data_top_10_percent(sample_data):
    response_df, predictors_df = sample_data
    y, X = generate_modeling_data(
        "tf1", response_df, predictors_df, quantile_threshold=0.1
    )
    assert isinstance(y, pd.DataFrame)
    assert isinstance(X, pd.DataFrame)
    assert y.shape[0] == X.shape[0]
    assert "tf1" in y.columns
    assert all(
        [col in X.columns for col in ["tf1", "tf1:tf2", "tf1:tf3"]]
    ), "The predictor columns are not as expected."
    assert y.shape[0] == 10


def test_generate_modeling_data_missing_col(sample_data):
    response_df, predictors_df = sample_data
    with pytest.raises(
        ValueError,
        match="The column nonexistent does not exist in the response DataFrame",
    ):
        generate_modeling_data("nonexistent", response_df, predictors_df)


# Tests for stratification_classification
def test_stratification_classification():
    binding_vector = pd.Series(np.random.randn(100))
    perturbation_vector = pd.Series(np.random.randn(100))
    classes = stratification_classification(binding_vector, perturbation_vector)
    assert isinstance(classes, np.ndarray)
    assert len(classes) == len(binding_vector)


def test_stratification_classification_mismatch_length():
    binding_vector = pd.Series(np.random.randn(100))
    perturbation_vector = pd.Series(np.random.randn(101))
    with pytest.raises(ValueError):
        stratification_classification(binding_vector, perturbation_vector)


# Tests for stratified_cv_modeling
def test_stratified_cv_modeling(sample_data):
    response_df, predictors_df = sample_data
    y, X = generate_modeling_data("tf1", response_df, predictors_df)
    model = stratified_cv_modeling(y, X, LassoCV())
    assert hasattr(model, "coef_")


# Tests for bootstrap_stratified_cv_modeling
def test_bootstrap_stratified_cv_modeling(lasso_output):
    ci_dict, bootstrap_coefs_df, alpha_list = lasso_output
    assert isinstance(ci_dict, dict)
    assert isinstance(bootstrap_coefs_df, pd.DataFrame)
    assert bootstrap_coefs_df.shape[0] == 10  # n_bootstraps


def test_bootstrap_stratified_cv_modeling_invalid_bootstraps(sample_data):
    response_df, predictors_df = sample_data
    y, X = generate_modeling_data("tf1", response_df, predictors_df)
    with pytest.raises(
        ValueError, match="The number of bootstraps must be an integer greater than 0"
    ):
        bootstrap_stratified_cv_modeling(n_bootstraps=0, y=y, X=X, estimator=LassoCV())


# Tests for examine_bootstrap_coefficients
def test_examine_bootstrap_coefficients(lasso_output):
    plt_obj, sig_coef = examine_bootstrap_coefficients(
        lasso_output, ci_level="95.0", threshold=0
    )
    assert isinstance(
        plt_obj, plt.Figure
    ), "The function should return a Matplotlib figure."
    plt.close(plt_obj)  # Close plot after test


def test_examine_bootstrap_coefficients_custom_threshold(lasso_output):
    plt_obj, sig_coef = examine_bootstrap_coefficients(
        lasso_output, ci_level="95.0", threshold=0.5
    )
    assert isinstance(
        plt_obj, plt.Figure
    ), "The function should return a Matplotlib figure with custom threshold."
    plt.close(plt_obj)
