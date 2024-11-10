import logging
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy as pt
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LassoCV
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample

logger = logging.getLogger("main")


def generate_modeling_data(
    colname: str,
    response_df: pd.DataFrame,
    predictors_df: pd.DataFrame,
    drop_intercept: bool = True,
    formula: str | None = None,
    quantile_threshold: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate the response and predictor data, optionally filtering to the top x
    quantile.

    :param colname: The column name to use as the response variable. This column name
        should exist in `response_df` and `predictors_df`.
    :param response_df: The transformed response variable DataFrame.
    :param predictors_df: The predictors DataFrame.
    :param drop_intercept: Whether to drop the intercept in the formula. This adds
        a -1 to the formula if True. See the patsy docs
        https://patsy.readthedocs.io/en/latest/formulas.html#intercept-handling.
    :param formula: The formula to use for the interaction model. If None, the formula
        will be generated automatically. The formula should be in the form of
        `colname_LRR ~ predictor1 + predictor2 + ... + predictorN`. If `drop_intercept`
        is True, a -1 will be added to the formula. Note that _LRR needs to be added
        to the response variable name in the formula.
    :param quantile_threshold: If specified, filters the data to only include rows
        where `predictors_df[colname]` is in the top `quantile_threshold` quantile.
        For example, `quantile_threshold=0.1` would include only the top 10%.

    :return: A tuple of the response variable DataFrame and the predictors DataFrame.

    :raises ValueError: If `colname` does not exist in `predictors_df`
        or `response_df`.
    :raises ValueError: If any columns in `response_df` are not present
        in `predictors_df`.

    """
    # Validate input
    if colname not in response_df.columns:
        raise ValueError(
            f"The column {colname} does not exist in the response DataFrame."
        )

    # Ensure all columns in response_df are in predictors_df (excluding rep patterns)
    # TODO: this is a hack. We are assuming that the response column might have
    # a _rep\d+, removing it, and using it as the column name. Somehow this needs to
    # be handled in a more robust way. See the other TODOS on this
    response_columns = response_df.columns.str.replace(r"_rep\d+", "", regex=True)
    missing_cols = response_columns.difference(predictors_df.columns)
    if not missing_cols.empty:
        raise ValueError(
            "The following columns are missing from the predictors DataFrame: "
            f"{missing_cols}"
        )

    # Merge response and predictors DataFrames on their index
    tmp_df = pd.merge(
        response_df[[colname]].rename(columns={colname: f"{colname}_LRR"}),
        predictors_df,
        left_index=True,
        right_index=True,
        how="inner",
    )

    # check if colname is in the index. If it is, drop it
    if colname not in tmp_df.index:
        logger.warning(
            f"{colname} is not in the index. The perturbed gene may "
            "not be removed from the modeling data"
        )
    else:
        logger.info(
            f"Removing {colname} from the data rows (removing the perturbed TF)"
        )
        tmp_df = tmp_df.drop(colname)

    # log the number of rows in the merged dataframe
    logger.info(f"Number of rows in the merged response/predictors: {tmp_df.shape[0]}")

    # remove the _rep\d+ pattern from the response colname -- perturbation data should
    # not have replicates.
    # TODO: this needs to be handled more transparently. See the other TODOs on
    # rep\d
    perturbed_tf = re.sub(r"_rep\d+", "", colname)

    # Apply quantile filtering if quantile_threshold is specified
    if quantile_threshold is not None:
        quantile_value = tmp_df[perturbed_tf].quantile(1 - quantile_threshold)
        tmp_df = tmp_df[tmp_df[perturbed_tf] >= quantile_value]
        logger.info(
            f"Number of rows after filtering by the top "
            f"{quantile_threshold} of {perturbed_tf}: {tmp_df.shape[0]}"
        )

    # Step 3: Define the interaction formula
    if formula is None:
        interaction_terms = " + ".join(
            [
                f"{perturbed_tf}:{other_col}"
                for other_col in predictors_df.columns
                if other_col != perturbed_tf
            ]
        )
        formula = f"{colname}_LRR ~ {perturbed_tf} + {interaction_terms}"

    if drop_intercept:
        formula += " - 1"

    # Step 4: Generate X, y matrices with patsy
    logger.info(f"Generating modeling data with formula: {formula}")
    y, X = pt.dmatrices(formula, tmp_df, return_type="dataframe")

    # Clean up column names in `y`
    y.columns = y.columns.str.replace("_LRR", "")

    return y, X


def stratification_classification(
    binding_series: pd.Series,
    perturbation_series: pd.Series,
    bins: list = [0, 8, 64, 512, np.inf],
) -> np.ndarray:
    """
    Bin the binding and perturbation data and create groups for stratified k folds.

    :param binding_series: The binding vector to use for stratification
    :param perturbation_series: The perturbation vector to use for stratification
    :param bins: The bins to use for stratification.
        The default is [0, 8, 64, 512, np.inf]

    :return: A numpy array of the stratified classes

    :raises ValueError: If the length of `binding_series` and
        `perturbation_series` are not equal
    :raises ValueError: If the length of `bins` is less than 2
    :raises ValueError: If `binding_series` and `perturbation_series` are not numeric
        pd.Series

    """
    # Validate input
    if len(binding_series) != len(perturbation_series):
        raise ValueError(
            "The length of the binding and perturbation vectors must be equal."
        )
    if len(bins) < 2:
        raise ValueError("The number of bins must be at least 2.")
    if not isinstance(binding_series, pd.Series) or not isinstance(
        perturbation_series, pd.Series
    ):
        raise ValueError("The binding and perturbation vectors must be pandas Series.")
    if (
        binding_series.dtype.kind not in "biufc"
        or perturbation_series.dtype.kind not in "biufc"
    ):
        raise ValueError("The binding and perturbation vectors must be numeric.")

    # Rank genes by binding and perturbation scores
    binding_rank = binding_series.rank(method="min", ascending=False).values
    perturbation_rank = perturbation_series.rank(method="min", ascending=False).values

    # Generate labels based on the number of bins
    # Number of labels is one less than the number of bin edges
    labels = list(range(1, len(bins)))

    # Bin genes based on binding and perturbation ranks
    binding_bin = pd.cut(binding_rank, bins=bins, labels=labels, right=True).astype(int)
    perturbation_bin = pd.cut(
        perturbation_rank, bins=bins, labels=labels, right=True
    ).astype(int)

    # Generate a combined classification based on the binding and perturbation bins
    # each combination of binding and perturbation bin will be a unique class
    return (binding_bin - 1) * len(labels) + perturbation_bin


def stratified_cv_modeling(
    y: pd.DataFrame,
    X: pd.DataFrame,
    estimator: BaseEstimator = LassoCV(),
    sample_weight: np.ndarray | None = None,
    drop_columns_before_modeling: list = [],
) -> BaseEstimator:
    """
    This conducts the LassoCV modeling. The name `stratified_cv_modeling` is a misnomer.
    There is nothing in this function that requires any specific model.

    :param y: The response variable to use for modeling. This should be a single column.
        See `generate_modeling_data()`
    :param X: The predictors to use for modeling. This should be an NxP DataFrame where
        N == len(y) and P is the number of predictors. See `generate_modeling_data()`
    :param estimator: The estimator to use for fitting the model. It must have a `cv`
        attribute that can be set with a list of StratifiedKFold splits
    :param sample_weight: The sample weights to use for fitting the model. Default is
        None, which is the default behavior for LassoCV.fit()

    :return: The LassoCV model

    :raises ValueError: if y is not a single column DataFrame
    :raises ValueError: if X is not a DataFrame with 1 or more columns, or the number
        of rows in y does not match the number of rows in X
    :raises ValueError: If the estimator does not have a `cv` attribute

    """
    # Validate data
    if not isinstance(y, pd.DataFrame):
        raise ValueError("The response variable y must be a DataFrame.")
    if y.shape[1] != 1:
        raise ValueError("The response variable y must be a single column DataFrame.")
    if not isinstance(X, pd.DataFrame):
        raise ValueError("The predictors X must be a DataFrame.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("The number of rows in X must match the number of rows in y.")

    # Verify estimator has a `cv` attribute
    if not hasattr(estimator, "cv"):
        raise ValueError("The estimator must support a `cv` parameter.")

    # Step 5: Generate bins for stratified k-fold cross-validation
    # TODO: This is a hack. We are assuming that the response column might have
    # a _rep\d+, removing it, and using it as the column name. Somehow this needs to
    # be handled in a more robust way.
    response_colname_no_rep = re.sub(r"_rep\d+", "", y.columns[0])
    if response_colname_no_rep not in X.columns:
        raise ValueError(
            f"The response column {response_colname_no_rep} does not exist "
            "in the predictors. This is currently expected in order to create the "
            "stratified folds. If different behavior is desired, the code will need "
            "to be modified."
        )
    classes = stratification_classification(X[response_colname_no_rep], y.squeeze())

    # Step 6: Initialize StratifiedKFold for stratified splits
    logger.debug("Generating stratified k-fold splits")
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        folds = list(skf.split(X, classes))
        for warning in w:
            logger.warning(
                f"Warning encountered during stratified k-fold split: {warning.message}"
            )

    # Clone the estimator and set the `cv` attribute with predefined folds
    model = clone(estimator)
    model.cv = folds

    # Step 7: Fit the model using the custom cross-validation folds
    logger.debug("Fitting the model")
    if drop_columns_before_modeling:
        logger.info(f"Dropping columns {drop_columns_before_modeling} before modeling")

    model.fit(
        X.drop(drop_columns_before_modeling, axis=1),
        y.values.ravel(),
        sample_weight=sample_weight,
    )

    return model


def bootstrap_stratified_cv_modeling(
    n_bootstraps: int = 1000,
    ci_percentiles: list[float] = [95.0, 99.0],
    use_sample_weight_in_cv: bool = False,
    **kwargs,
) -> tuple[dict[str, dict[str, tuple[float, float]]], pd.DataFrame, list[float]]:
    """
    Perform bootstrap resampling to generate confidence intervals for
    Lasso coefficients. See 6.2 in https://hastie.su.domains/StatLearnSparsity/ -- this
    is an implementation of the algorithm described in that section.

    :param n_bootstraps: The number of bootstrap samples to generate.
    :param ci_percentiles: A list of CI percentiles to calculate. Default to [95, 99]
    :param use_sample_weight_in_cv: Whether to use sample weights, calculated as the
        proportion of times a given record appears, in boostrap iterations. Default is
        False.
    :param kwargs: The required arguments to `stratified_cv_modeling` must be
        passed as keyword arguments to this function. Any additional keyword arguments
        will be passed to the Lasso estimator.

    :return: A tuple where:
        - The first element is a dictionary of confidence intervals,
          where keys are CI levels (e.g., "95.0") and values are dictionaries
          mapping each coefficient to its lower and upper bounds, with columns named
          according to the predictors in `X`.
        - The second element is a DataFrame where each row represents a
        bootstrap sample,
          and columns correspond to the predictor variables from `X`.
        - The third element is a list of alpha values from each bootstrap iteration.


    :raises ValueError: If any of the required keyword arguments for
        `stratified_cv_modeling()` are not passed.
    :raises ValueError: If `n_bootstraps` is not an integer greater than 0.
    :raises ValueError: If `ci_percentiles` is not a list of integers or floats.
    :raises ValueError: If `use_sample_weight_in_cv` is not a boolean.
    """
    # Validate input parameters
    if not isinstance(n_bootstraps, int) or n_bootstraps < 1:
        raise ValueError("The number of bootstraps must be an integer greater than 0.")
    if not isinstance(ci_percentiles, list) or not all(
        isinstance(x, (int, float)) for x in ci_percentiles
    ):
        raise ValueError(
            "The confidence interval percentiles must be a list of integers or floats."
        )
    if not isinstance(use_sample_weight_in_cv, bool):
        raise ValueError("The use_sample_weight_in_cv parameter must be a boolean.")

    # Extract and validate required arguments for modeling
    y = kwargs.pop("y")
    X = kwargs.pop("X")
    estimator = kwargs.pop("estimator")
    if any(x is None for x in [y, X, estimator]):
        raise ValueError(
            "Arguments 'y', 'X', and 'estimator' must be "
            "passed to bootstrap_cv_modeling()."
        )

    bootstrap_coefs = []
    alpha_list = []

    # Bootstrap iterations
    for _ in range(n_bootstraps):
        Y_resampled = resample(y, replace=True)
        X_resampled = X.loc[Y_resampled.index]

        weights = None
        if use_sample_weight_in_cv:
            index_mapping = {label: idx for idx, label in enumerate(y.index)}
            integer_indices = [index_mapping[label] for label in Y_resampled.index]
            sample_counts = np.bincount(integer_indices, minlength=len(y))
            weights = sample_counts / len(y)

        model_i = stratified_cv_modeling(
            Y_resampled,
            X_resampled,
            estimator,
            sample_weight=weights,
            drop_columns_before_modeling=kwargs.get("drop_columns_before_modeling", []),
        )
        alpha_list.append(model_i.alpha_)
        bootstrap_coefs.append(model_i.coef_)

    # Convert coefficients list to a DataFrame with column names from X
    bootstrap_coefs_df = pd.DataFrame(
        bootstrap_coefs,
        columns=X.drop(kwargs.get("drop_columns_before_modeling", []), axis=1).columns,
    )

    # Compute confidence intervals
    ci_dict = {
        f"{str(ci)}": {
            colname: (
                np.percentile(bootstrap_coefs_df[colname], (100 - ci) / 2),
                np.percentile(bootstrap_coefs_df[colname], 100 - (100 - ci) / 2),
            )
            for colname in bootstrap_coefs_df.columns
        }
        for ci in ci_percentiles
    }

    return ci_dict, bootstrap_coefs_df, alpha_list


def examine_bootstrap_coefficients(
    lasso_model_output, ci_level: str = "95.0", threshold: float = 0.0
) -> tuple[Figure, dict[str, tuple[float, float]]]:
    """
    Generate a plot, and output the set of coefficients that meet the ci_level and
    threshold criteria.

    Usage:

    Examples
    --------
    Display the plot immediately:

    ```python
    fig, significant_cols = plot_significant_lasso_coefficients(
        lasso_model_output,
        ci_level="95.0",
        threshold=0.5
    )
    fig.show()
    ```

    Further customize the plot:

    ```python
    fig, significant_cols = plot_significant_lasso_coefficients(
        lasso_model_output,
        ci_level="95.0",
        threshold=0.5
    )
    ax = fig.gca()  # Get the current axes
    ax.set_title("Custom Title for Lasso Coefficients", fontsize=16)
    ax.set_ylabel("Custom Y-axis Label")
    fig.show()
    ```

    Save the plot to a file:

    ```python
    fig, significant_cols = plot_significant_lasso_coefficients(
        lasso_model_output,
        ci_level="95.0",
        threshold=0.5
    )
    fig.savefig("significant_lasso_coefficients.png", dpi=300, bbox_inches='tight')
    ```

    Embed in Jupyter Notebooks:

    ```python
    plot_significant_lasso_coefficients(
        lasso_model_output,
        ci_level="95.0",
        threshold=0.5
    )
    ```

    Add annotations or modify appearance:

    ```python
    fig, significant_cols = plot_significant_lasso_coefficients(
        lasso_model_output,
        ci_level="95.0",
        threshold=0.5
    )
    ax = fig.gca()
    ax.annotate("Important Coefficient", xy=(1.5, 1), xytext=(2, 2),
                arrowprops=dict(facecolor='black', shrink=0.05))
    fig.show()
    ```

    :param lasso_model_output: The output from `bootstrap_cv_modeling()`.
        Expected format: (fitted_model, ci_dict, bootstrap_coefs_df, alpha_list)
    :param ci_level: The confidence interval level to use, e.g., "95.0".
    :param threshold: The threshold for selecting coefficients to plot.
        Only coefficients with confidence intervals entirely
        above or below this threshold will be displayed.

    :return: A tuple containing:
        - The created Matplotlib Figure for further customization.
        - A dictionary where the keys are the significant coefficients and the values
        are the confidence intervals specified in the `ci_level` parameter.

    """
    # Unpack lasso_model_output
    ci_dict, bootstrap_coefs_df, _ = lasso_model_output

    ci_dict_local = ci_dict.copy()

    # if ci_level is not in ci_dict, then it needs to be calculated from the
    # boostrap_coef_df
    if ci_level not in ci_dict_local:
        ci_dict_local[ci_level] = {
            colname: (
                np.percentile(bootstrap_coefs_df[colname], (100 - float(ci_level)) / 2),
                np.percentile(
                    bootstrap_coefs_df[colname], 100 - (100 - float(ci_level)) / 2
                ),
            )
            for colname in bootstrap_coefs_df.columns
        }

    # Identify coefficients with confidence intervals entirely above or
    # below the threshold, with a minimum distance from zero.
    significant_coefs_dict = {
        coef: bounds
        for coef, bounds in ci_dict_local[ci_level].items()
        if bounds[0] > threshold or bounds[1] < -threshold
    }

    # Display selected coefficients and their intervals
    print(
        f"Significant coefficients for {ci_level}, "
        f"where intervals are entirely above or below ±{threshold}:"
    )
    for coef, bounds in significant_coefs_dict.items():
        print(f"{coef}: {bounds}")

    # Extract the selected columns from bootstrap_coefs_df for plotting
    df_extracted = bootstrap_coefs_df[significant_coefs_dict.keys()]

    # Plotting the boxplot of the selected coefficients
    fig = plt.figure(figsize=(10, 6))

    # Add subtitle details about the threshold
    title_text = f"Coefficients with {ci_level}% CI intervals outside ±{threshold}"

    sns.boxplot(data=df_extracted, orient="h")
    plt.axvline(x=0, linestyle="--", color="black")
    plt.xlabel("Coefficient Values")

    # Main title and subtitle
    plt.title(title_text)

    return fig, significant_coefs_dict
