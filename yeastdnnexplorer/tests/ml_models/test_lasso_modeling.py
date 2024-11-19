import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LassoCV

# Import the functions from the module being tested
from yeastdnnexplorer.ml_models.lasso_modeling import (
    backwards_OLS_feature_selection,
    bootstrap_stratified_cv_modeling,
    examine_bootstrap_coefficients,
    generate_modeling_data,
    get_interactor_importance,
    select_significant_features,
    stratification_classification,
    stratified_cv_modeling,
    stratified_cv_r2,
    try_interactor_variants,
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
    classes = stratification_classification(
        predictors_df["tf1"].squeeze(), response_df.squeeze()
    )
    y, X = generate_modeling_data("tf1", response_df, predictors_df)
    model = stratified_cv_modeling(y, X, classes=classes, estimator=LassoCV())
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


# test for select_significant_features
def test_select_significant_features(sample_data):
    response_df, predictors_df = sample_data
    y, X = generate_modeling_data(
        "tf1",
        response_df,
        predictors_df,
        quantile_threshold=None,
        drop_intercept=True,
    )

    # Combine y and X into a single DataFrame for Patsy
    y = y.add_suffix("_LRR")
    feature_set = {"tf1", "tf1:tf2", "tf1:tf3"}
    significant_features = select_significant_features(feature_set, y, X, 0.05)
    assert isinstance(significant_features, list), "The output should be a list."
    assert all(
        isinstance(feature, str) for feature in significant_features
    ), "All features should be strings."


# test for backwards_OLS_Feature_selection
def test_backwards_OLS_feature_selection(sample_data):
    response_df, predictors_df = sample_data
    intersect_coefficients = {"tf1", "tf1:tf2", "tf1:tf3"}
    final_features = backwards_OLS_feature_selection(
        "tf1", intersect_coefficients, response_df, predictors_df, [None], [0.05]
    )
    assert (
        final_features or len(final_features) == 0
    ), "The set can be empty or have valid features."


# test for get_interactor_importance
def test_get_interactor_importance(sample_data):
    response_df, predictors_df = sample_data
    intersect_coefficients = {"tf1", "tf1:tf2"}
    main_effects = []
    for term in intersect_coefficients:
        if ":" in term:
            main_effects.append(term.split(":")[1])
        else:
            main_effects.append(term)
    interactor_terms_and_main_effects = list(intersect_coefficients) + main_effects
    y, X = generate_modeling_data(
        "tf1",
        response_df,
        predictors_df,
        formula=f"tf1_LRR ~ {' + '.join(interactor_terms_and_main_effects)}",
    )
    classes = stratification_classification(X["tf1"], y["tf1"])
    avg_r2, results = get_interactor_importance(y, X, classes, intersect_coefficients)
    assert isinstance(avg_r2, float), "The avg_r2 should be a float."
    assert isinstance(results, list), "Results should be a list."
    for result in results:
        assert isinstance(result, dict), "Each result should be a dictionary."
        assert "interactor" in result
        assert "variant" in result
        assert "avg_r2" in result


# test for try_interactor_variants
def test_try_interactor_variants(sample_data):
    response_df, predictors_df = sample_data
    intersect_coefficients = {"tf1:tf2", "tf1:tf3"}
    main_effects = []
    for term in intersect_coefficients:
        if ":" in term:
            main_effects.append(term.split(":")[1])
        else:
            main_effects.append(term)
    interactor_terms_and_main_effects = list(intersect_coefficients) + main_effects
    y, X = generate_modeling_data(
        "tf1",
        response_df,
        predictors_df,
        formula=f"tf1_LRR ~ tf1 +{' + '.join(interactor_terms_and_main_effects)}",
    )
    classes = stratification_classification(X["tf1"], y["tf1"])
    output = try_interactor_variants(
        intersect_coefficients,
        "tf1:tf2",
        y=y,
        X=X,
        stratification_classes=classes,
    )
    assert isinstance(output, list), "The output should be a list."
    assert all(isinstance(o, dict) for o in output), "Each item should be a dictionary."
    for result in output:
        assert "interactor" in result
        assert "variant" in result
        assert "avg_r2" in result
        assert isinstance(result["avg_r2"], float), "avg_r2 should be a float."


# test for stratified_cv_r2
def test_stratified_cv_r2(sample_data):
    response_df, predictors_df = sample_data
    y, X = generate_modeling_data("tf1", response_df, predictors_df)
    classes = stratification_classification(X["tf1"], y["tf1"])
    r2 = stratified_cv_r2(y, X, classes)
    assert isinstance(r2, float), "The result should be a float."
    assert -1 <= r2 <= 1, "R² should be between 0 and 1."
