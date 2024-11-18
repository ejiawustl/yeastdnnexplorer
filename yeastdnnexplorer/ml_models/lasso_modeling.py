import logging
import re
import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy as pt
import seaborn as sns
import statsmodels.api as sm
from matplotlib.figure import Figure
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import r2_score
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
    try:
        # Attempt to create matrices
        y, X = pt.dmatrices(formula, tmp_df, return_type="dataframe")
        # Clean up column names in `y`
        y.columns = y.columns.str.replace("_LRR", "")
    except pt.PatsyError as exc:
        # Check if it's the specific missing outcome variable error
        if "model is missing required outcome variables" in str(exc):
            logger.info("No outcome variable found in the formula")
            X = pt.dmatrix(formula, tmp_df, return_type="dataframe")
            y = pd.Series(dtype="float64")
        else:
            # Re-raise the error if it's something else
            raise

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
    classes: np.ndarray,
    estimator: BaseEstimator = LassoCV(),
    skf: StratifiedKFold = StratifiedKFold(n_splits=4, shuffle=True, random_state=42),
    sample_weight: np.ndarray | None = None,
) -> BaseEstimator:
    """
    This conducts the LassoCV modeling. The name `stratified_cv_modeling` is a misnomer.
    There is nothing in this function that requires any specific model.

    :param y: The response variable to use for modeling. This should be a single column.
        See `generate_modeling_data()`
    :param X: The predictors to use for modeling. This should be an NxP DataFrame where
        N == len(y) and P is the number of predictors. See `generate_modeling_data()`
    :param classes: The classes to use for stratified k-fold cross-validation. This
        should be an array of integers generated by `stratification_classification()`
    :param estimator: The estimator to use for fitting the model. It must have a `cv`
        attribute that can be set with a list of StratifiedKFold splits
    :param skf: The StratifiedKFold object to use for stratified splits. Default is
        StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    :param sample_weight: The sample weights to use for fitting the model. Default is
        None, which is the default behavior for LassoCV.fit()

    :return: The LassoCV model

    :raises ValueError: if y is not a single column DataFrame
    :raises ValueError: if X is not a DataFrame with 1 or more columns, or the number
        of rows in y does not match the number of rows in X
    :raises ValueError: if classes is not a numpy array or is empty
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
    if classes.size == 0 or not isinstance(classes, np.ndarray):
        raise ValueError("The classes must be a non-empty numpy array.")

    # Verify estimator has a `cv` attribute
    if not hasattr(estimator, "cv"):
        raise ValueError("The estimator must support a `cv` parameter.")

    # Initialize StratifiedKFold for stratified splits
    logger.debug("Generating stratified k-fold splits")
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
    model.fit(
        X,
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
    r"""
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
    :raises ValueError: If the response variable is not in the predictors DataFrame.
        If there are replicates, they are expected to have the suffix _rep\d+. This
        is attempted to be removed from the response variable name to match the
        predictors DataFrame.
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

    response_tf = re.sub(r"_rep\d+", "", y.columns[0])

    if response_tf not in X.columns:
        raise ValueError(
            f"The response variable {response_tf} is not in the predictors DataFrame."
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

        classes = stratification_classification(
            X_resampled.loc[:, response_tf].squeeze(), Y_resampled.squeeze()
        )

        model_i = stratified_cv_modeling(
            Y_resampled,
            X_resampled,
            classes=classes,
            estimator=estimator,
            sample_weight=weights,
        )
        alpha_list.append(model_i.alpha_)
        bootstrap_coefs.append(model_i.coef_)

    # Convert coefficients list to a DataFrame with column names from X
    bootstrap_coefs_df = pd.DataFrame(
        bootstrap_coefs,
        columns=X.columns,
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


# TODO: must have option to add max_lrb

# TODO: the top10% models likely should not have teh same number of classes as the
# all data, possibly not stratified at all


def get_significant_predictors(
    perturbed_tf: str,
    response_df: pd.DataFrame,
    predictors_df: pd.DataFrame,
    add_max_lrb: bool,
    **kwargs: Any,
) -> tuple[dict[str, tuple[float, float]], pd.DataFrame, pd.Series]:
    """
    This function is used to get the significant predictors for a given TF. It is
    capable of conducting steps 1.1 and 1.2 described above.

    :param perturbed_tf: the TF for which the significant predictors are to be
        identified
    :param response_df: The DataFrame containing the response values
    :param predictors_df: The DataFrame containing the predictor values
    :param add_max_lrb: A boolean to add/not add in the max_LRB term for a response TF
        into the formula that we perform bootstrapping on
    :param kwargs: Additional arguments to be passed to the function. Expected arguments
        are 'quantile_threshold' from generate_modeling_data() and 'ci_percentile' from
        examine_bootstrap_coefficients()
    :return sig_coef_dict: A dictionary containing the significant predictors and their
        corresponding coefficients

    """

    y, X = generate_modeling_data(
        perturbed_tf,
        response_df,
        predictors_df,
        quantile_threshold=kwargs.get("quantile_threshold", None),
        drop_intercept=True,
    )

    # NOTE: fit_intercept is set to `true`
    lassoCV_estimator = LassoCV(
        fit_intercept=True,
        max_iter=10000,
        selection="random",
        random_state=42,
        n_jobs=4,
    )

    predictor_variable = re.sub(r"_rep\d+", "", perturbed_tf)

    stratification_classes = stratification_classification(
        X[predictor_variable].squeeze(), y.squeeze()
    )

    # Fit the model to the data in order to extract the alphas_ which are generated
    # during the fitting process
    lasso_model = stratified_cv_modeling(
        y, X, stratification_classes, lassoCV_estimator
    )

    if add_max_lrb:
        # add a column to X which is the rowMax excluding column `predictor_variable`
        # called max_lrb
        X["max_lrb"] = X.drop(columns=predictor_variable).max(axis=1)

    # set the alphas_ attribute of the lassoCV_estimator to the alphas_ attribute of the
    # lasso_model fit on the whole data. This will allow the
    # bootstrap_stratified_cv_modeling function to use the same set of lambdas
    lassoCV_estimator.alphas_ = lasso_model.alphas_

    # for test purposes, set n_bootstraps to 10
    # NOTE: fit_intercept=True is passed to the internal Lasso model for bootstrap
    # iterations, along with some other settings

    logging.info("running bootstraps")
    bootstrap_lasso_output = bootstrap_stratified_cv_modeling(
        y=y,
        X=X,
        estimator=lassoCV_estimator,
        ci_percentile=kwargs.get("ci_percentile", 95.0),
        n_bootstraps=kwargs.get("n_bootstraps", 10),
        max_iter=10000,
        fit_intercept=True,
        selection="random",
        random_state=42,
    )

    sig_coef_plt, sig_coef_dict = examine_bootstrap_coefficients(
        bootstrap_lasso_output, ci_level=kwargs.get("ci_percentile", 95.0)
    )

    plt.close(sig_coef_plt)

    return sig_coef_dict, y, stratification_classes


def stratified_cv_r2(
    y: pd.DataFrame,
    X: pd.DataFrame,
    classes: np.ndarray,
    estimator: BaseEstimator = LinearRegression(),
    skf: StratifiedKFold = StratifiedKFold(n_splits=4, shuffle=True, random_state=42),
) -> float:
    """
    Calculate the average stratified CV r-squared for a given estimator and data. By
    default, this is a 4-fold stratified CV with a LinearRegression estimator.

    :param y: The response variable. See generate_modeling_data()
    :param X: The predictor variables. See generate_modeling_data()
    :param classes: the stratification classes for the data
    :param estimator: the estimator to be used in the modeling. By default, this is a
        LinearRegression() model.
    :param skf: the StratifiedKFold object to be used in the modeling. By default, this
        is a 4-fold stratified CV with shuffle=True and random_state=42.
    :return: the average r-squared value for the stratified CV

    """

    estimator_local = clone(estimator)
    r2_scores = []

    for train_idx, test_idx in skf.split(X, classes):
        # Use train and test indices to split X and y
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Fit the model
        model = estimator_local.fit(X_train, y_train)

        # Calculate R-squared and append to r2_scores
        r2_scores.append(r2_score(y_test, model.predict(X_test)))

    return np.mean(r2_scores)


def try_interactor_variants(
    intersect_coefficients, interactor, **kwargs
) -> list[dict[str, Any]]:
    """
    For a given interactor, replace the term in the formula with one variant:
        1. the main effect
        IGNORE FOR NOW: 2. the main effect + interactor
    For each of the variants, calculate the average stratified CV r-squared with
    stratified_cv_r2().

    :param intersect_coefficients: the set of coefficients that are determined to be
        significant, expected to be from either a bootstrap procedure on a LassoCV
        model on a full partition of the data and the top 10% by perturbed binding, or
        LassoCV followed by backwards selection by adj-rsquared.
    :param interactor: the interactor term to be tested
    :param kwargs: additional arguments to be passed to the function. Expected
        arguments are 'y', 'X', and 'stratification_classes'. See stratified_cv_r2()
        for more information.

    :return: a list with three dict entries, each with key
        'interactor', 'variant', 'avg_r2'
    """
    y = kwargs.get("y")
    if y is None:
        raise ValueError("y must be passed as a keyword argument")
    X = kwargs.get("X")
    if X is None:
        raise ValueError("X must be passed as a keyword argument")
    stratification_classes = kwargs.get("stratification_classes")
    if stratification_classes is None:
        raise ValueError("stratification_classes must be passed as a keyword argument")

    main_effect = interactor.split(":")[1]
    # interactor_formula_variants = [main_effect, [main_effect, interactor]]

    # NOTE: may need to change this back to the original line above: I just noticed in
    # the Yeast To-Dos document that in the final pipeline, we only compare our formula
    # to the variant in which the main effect is subbed in. It appears we do not try
    # the formula in which we also add the main effect while keeping the interaction
    interactor_formula_variants = [main_effect]

    output = []
    for variant in interactor_formula_variants:
        # replace the interactor term in the formula with the variant
        variant_predictors = (
            [term for term in intersect_coefficients if term != interactor] + [variant]
            if isinstance(variant, str)
            else variant
        )
        # conduct the stratified CV r-squared calculation with the formula variant
        input_model_avg_rsquared = stratified_cv_r2(
            y, X.loc[:, variant_predictors], stratification_classes
        )

        # append the results to the output list
        output.append(
            {
                "interactor": interactor,
                "variant": variant,
                "avg_r2": input_model_avg_rsquared,
            }
        )
    return output


def get_interactor_importance(
    y: pd.DataFrame,
    full_X: pd.DataFrame,
    stratification_classes: np.ndarray,
    intersect_coefficients: set,
) -> tuple[float, list[dict[str, Any]]]:
    """
    For each interactor in the intersect_coefficients, run test_interactor_importance to
    compare the variants' avg_rsquared to the input_model_avg_rsquared. If a variant of
    the interactor term is better, record it in a dictionary. Return the
    `instersect_coefficient` model's avg R-squared and the dictionary of interaction
    alternatives that, when that alternative replaces a single interaction term,
    improves the rsquared.

    :param y: the response variable
    :param full_X: the full predictor matrix
    :param stratification_classes: the stratification classes for the data
    :param intersect_coefficients: the set of coefficients that are determined to be
        significant, expected to be from either a bootstrap procedure on a LassoCV model
        on a full partition of the data and the top 10% by perturbed binding, or LassoCV
        followed by backwards selection by p-value significance.
    :return: a tuple with the first element being the input_model_avg_rsquared and the
        second element being a list of dictionaries with keys 'interactor', 'variant',
        and 'avg_r2'

    """

    input_model_avg_rsquared = stratified_cv_r2(
        y, full_X.loc[:, list(intersect_coefficients)], stratification_classes
    )

    # for each interactor in the intersect_coefficients, run test_interactor_importance
    # compare the variants' avg_rsquared to the input_model_avg_rsquared. Record
    # the best performing.
    interactor_results = []
    for interactor in intersect_coefficients:
        if ":" in interactor:

            interactor_variant_results = try_interactor_variants(
                intersect_coefficients,
                interactor,
                y=y,
                X=full_X,
                stratification_classes=stratification_classes,
            )

            # compare the avg_r2 values of the two variants to input_model_avg_rsquared
            variant_dict = max(interactor_variant_results, key=lambda x: x["avg_r2"])

            if variant_dict["avg_r2"] > input_model_avg_rsquared:
                interactor_results.append(variant_dict)

    return input_model_avg_rsquared, interactor_results


def get_non_zero_predictors(
    perturbed_tf: str,
    response_df: pd.DataFrame,
    predictors_df: pd.DataFrame,
    **kwargs: Any,
) -> list[str]:
    """
    This function is used to get the features with non-zero coefficients from LassoCV
    for a given TF. It is capable of conducting steps 1a and 1b above.

    :param perturbed_tf: str, the TF for which the significant predictors are to be
        identified
    :param response_df: pd.DataFrame, the response dataframe containing the response
        values
    :param predictors_df: pd.DataFrame, the predictors dataframe containing the
        predictor values
    :param kwargs: dict, additional arguments to be passed to the function. Expected
        arguments is 'quantile_threshold' fom generate_modeling_data() to signify the
        top 10%
    :return non_zero_features: List, a list containing the features with non-zero coefs

    """
    # Validate input
    if perturbed_tf not in response_df.columns:
        raise ValueError(
            f"The response TF {perturbed_tf} does not exist in the response DataFrame."
        )
    if perturbed_tf not in predictors_df.columns:
        raise ValueError(
            f"The response TF {perturbed_tf} does not exist in the binding DataFrame."
        )
    if response_df.shape[0] != predictors_df.shape[0]:
        raise ValueError(
            "The binding and response DataFrames contain different counts of genes"
        )

    tf_y, tf_X = generate_modeling_data(
        perturbed_tf,
        response_df,
        predictors_df,
        quantile_threshold=kwargs.get("quantile_threshold", None),
        drop_intercept=True,
    )

    # add the max_lrb term to the formula
    tf_X["max_lrb"] = predictors_df.drop(columns=perturbed_tf).max(axis=1)

    # NOTE: fit_intercept is set to `true`
    lassoCV_estimator = LassoCV(
        fit_intercept=True,
        max_iter=10000,
        selection="random",
        random_state=42,
        n_jobs=4,
    )

    # create the classifications
    classes = stratification_classification(tf_X[perturbed_tf], tf_y[perturbed_tf])

    # Fit the model to the data in order to extract the non-zero coefficients
    # surviving after the fitting process
    lasso_model = stratified_cv_modeling(tf_y, tf_X, classes, lassoCV_estimator)

    # return a list of the non-zero features that survived the fitting
    non_zero_indices = lasso_model.coef_ != 0
    non_zero_features = tf_X.columns[non_zero_indices]

    return non_zero_features


def backwards_OLS_feature_selection(
    perturbed_tf: str,
    intersect_coefficients: set[str],
    response_df: pd.DataFrame,
    predictors_df: pd.DataFrame,
) -> set[str]:
    """
    Perform backward feature selection using OLS to iteratively filter down a set of
    input features. This is the wrapper method that performs Step 3 of Workflow 2 in the
    interactor_modeling_workflow notebook. It takes in the intersected coefficients from
    Step 2 and calls the helper method get_full_data() and select_significant_features()
    in order to continuously update the set of features by eliminating those that are
    above a given p-value threshold. It does this on the whole data first, then the top
    10% of the data by perturbed TF binding until there are no more features that are
    insignificant. It returns this set of features.

    :param perturbed_tf: The name of the response TF.
    :param intersect_coefficients: The initial intersected set of predictor features
        passed in from Step 2.
    :param predictors_df: A DataFrame containing the predictor data for the response TF.
    :param response_df: A DataFrame containing the response data for the response TF.
    :return: The final refined set of significant predictors.

    """
    # Get the full dataset
    full_data = get_full_data(
        perturbed_tf,
        response_df,
        predictors_df,
        quantile_threshold=None,  # Full dataset has no quantile filtering
    )

    # Get the top 10% binding dataset
    top_10_data = get_full_data(
        perturbed_tf, response_df, predictors_df, quantile_threshold=0.1
    )

    # keep count of the feature set size to  track when to terminate the while loop
    prev_set_size = 0
    curr_set_size = len(intersect_coefficients)
    curr_feature_set = intersect_coefficients

    # Step 3.1: Iteratively filter features using the full dataset
    while curr_set_size != prev_set_size and curr_set_size != 0:
        curr_feature_set = set(
            select_significant_features(
                perturbed_tf, curr_feature_set, full_data, p_value_threshold=0.001
            )
        )
        prev_set_size = curr_set_size
        curr_set_size = len(curr_feature_set)

    # Reinitialize variables for the second loop
    prev_set_size = 0  # Reset for Step 3.2
    curr_set_size = len(curr_feature_set)

    # Step 3.2: Iteratively filter features using the top 10% dataset
    # this time we use a less stringent p-value threshold since there is less data
    while curr_set_size != prev_set_size and curr_set_size != 0:
        curr_feature_set = set(
            select_significant_features(
                perturbed_tf, curr_feature_set, top_10_data, p_value_threshold=0.01
            )
        )
        prev_set_size = curr_set_size
        curr_set_size = len(curr_feature_set)

    return curr_feature_set


def get_full_data(
    perturbed_tf: str,
    response_df: pd.DataFrame,
    predictors_df: pd.DataFrame,
    quantile_threshold: float | None = None,
) -> pd.DataFrame:
    """
    This is a helper method for backwards_OLS_feature_selection and is a wrapper of
    generate_modeling_data() which transforms the y and X matrices output from that
    function into a single DataFrame that is compatible with Patsy for creating OLS
    models for backward OLS feature selection.

    :param perturbed_tf: The name of the response TF.
    :param response_df: A DataFrame containing the response data.
    :param predictors_df: A DataFrame containing the predictor data.
    :param quantile_threshold: Optional float to filter data on quantile thresholds.
        This is used for getting the top 10% of the data in Step 3.2 of Workflow 2.
    :return: A DataFrame combining the response and predictor variable columns.

    """
    # Generate response (y) and predictors (X) using generate_modeling_data
    y, X = generate_modeling_data(
        perturbed_tf,
        response_df,
        predictors_df,
        drop_intercept=True,
        quantile_threshold=quantile_threshold,
    )

    # Rename response column to append "_LRR" (for Patsy to differentiate between
    # response TF and its main effect columns)
    y = y.rename(columns=lambda col: col + "_LRR")

    # Concatenate response and predictor data
    full_data = pd.concat([y, X], axis=1, join="inner")

    # add the max_lrb column
    full_data["max_lrb"] = predictors_df.drop(columns=perturbed_tf).max(axis=1)

    return full_data


def select_significant_features(
    perturbed_tf: str, feature_set: set, data: pd.DataFrame, p_value_threshold: float
) -> list:
    """
    This is a helper method for backwards_OLS_feature_selection which select significant
    features from a given set of predictors by creating an OLS model and removing
    features whose p-values are above a given threshold. This performs a single
    iteration of filtering, and returns the set of interactors that survive after.

    Note: because the data input into this method was created by get_full_data(), which
    is a wrapper of generate_modeling_data(), the columns already are the interactions
    between TFs. This is an issue with Patsy because when it sees "TF1:TF2" as an
    example feature in the formula whose design matrix it is creating, it thinks to
    find the individual columns "TF1" and "TF2" and taking their product. However, since
    the data passed in is only of the form "TF1:TF2" for all interactions, we need to
    temporarily swap the ":" for an "_" so that Patsy doesn't generate an error saying
    it cannot identify the data needed.

    :param perturbed_tf: The name of the response TF.
    :param feature_set: Set of predictor features to be filtered on. This might be the
        initial set of intersected features passed in from Step 2, or any subsequent
        filtered set of features output from previous iterations of calling this method.
    :param data: A DataFrame containing both predictors and response data for Patsy.
    :param p_value_threshold: A threshold for qualifying significance for features.

    :return: List of significant predictors with original names.

    """
    # Create a mapping of original names to modified names - see docstring for details
    name_mapping = {col: col.replace(":", "_") for col in feature_set}

    # Replace colons with underscores in feature_set and data column names
    modified_feature_set = {name_mapping[col] for col in feature_set}
    data.columns = data.columns.str.replace(":", "_")

    # Generate formula
    predictors_str = " + ".join(modified_feature_set)
    response = f"{perturbed_tf}_LRR"
    formula = f"{response} ~ {predictors_str}"

    # Generate the design matrix
    y, X = pt.dmatrices(formula, data, return_type="dataframe")

    # Fit the OLS model and identify significant features
    model = sm.OLS(y, X).fit()
    significant_features = [
        feature
        for feature, pval in model.pvalues.items()
        if pval < p_value_threshold and feature != "Intercept"
    ]

    # Convert modified names back to their original names
    reverse_mapping = {v: k for k, v in name_mapping.items()}
    significant_features = [
        reverse_mapping.get(feature, feature) for feature in significant_features
    ]

    return significant_features
