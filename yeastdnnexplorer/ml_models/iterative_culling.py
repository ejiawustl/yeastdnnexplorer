import logging
from typing import Any

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from patsy import dmatrices
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger("model_selection")


def create_formulas_with_max_lrb(tf_list):
    max_formulas = []
    for tf in tf_list:
        # Create the response variable
        response = f"LRR_{tf}"

        # Create the main effect for the current TF
        main_effect = f"LRB_{tf}"

        # Create the interaction terms with all other TFs
        interaction_terms = " + ".join(
            [f"LRB_{tf}:LRB_{other_tf}" for other_tf in tf_list if other_tf != tf]
        )

        # Combine to form the formula
        formula = f"{response} ~ {main_effect}"
        if interaction_terms:  # Add interaction terms if there are any
            formula += " + " + interaction_terms
        formula += f"+ max_LRB_{tf}"
        # Add the formula to the list
        max_formulas.append(formula)
    return max_formulas


def process_tf_data_combined(
    tf_name,
    data,
    formulas,
    filter_top_10=False,
    quantile_threshold=0.9,
    n_splits=4,
    random_state=42,
):
    # Find the formula corresponding to the response TF
    formula = [s for s in formulas if s.startswith(f"LRR_{tf_name}")]
    if not formula:
        raise ValueError(f"No formula found for {tf_name}")

    # Filter to use the top10% by binding data if indicated
    if filter_top_10:
        top_10_percent_threshold = data[f"LRB_{tf_name}"].quantile(quantile_threshold)
        data = data[data[f"LRB_{tf_name}"] >= top_10_percent_threshold]

    # Add classification to each gene for CV
    cv_data = classify_genes(data, tf_name)

    # Create a patsy dmatrix - note that we don't use an intercept as we will
    # specify that in the LassoCV model
    y, X = dmatrices(formula[0] + " - 1", data, return_type="dataframe")

    # Make splits for CV
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = list(
        skf.split(cv_data["target_locus_tag"].values, cv_data["classification"])
    )

    # Create the LassoCV model with an intercept and assign splits
    lasso_cv = LassoCV(cv=splits, max_iter=100000, fit_intercept=True)
    lasso_cv.fit(X, y)

    # Find only the non-zero coefficients
    non_zero_indices = lasso_cv.coef_ != 0
    feature_names = X.columns[non_zero_indices]
    non_zero_coefs = lasso_cv.coef_[non_zero_indices]

    results_df = pd.DataFrame({"Feature": feature_names, "Coefficient": non_zero_coefs})

    return results_df


def find_common_features_across_tfs(tfs, data, formulas, quantile_threshold=0.9):
    """
    Find common features across TFs by processing both the full dataset and the top 10%
    data.

    Parameters:
    tfs (list): List of transcription factors (TFs) to process.
    data (DataFrame): The DataFrame containing the data for processing.
    formulas (list): A list of formulas for different TFs.
    quantile_threshold (float, optional): The quantile threshold for top 10% filtering.

    Returns:
    dict: A dictionary where the key is a TF and the value is a list of common features.

    """
    common_features_dict: dict[str, list[str]] = {}

    for tf in tfs:
        # Perform Lasso Regression on the full dataset
        df_all_data = process_tf_data_combined(tf, data, formulas, filter_top_10=False)

        # Perform Lasso Regression on only the top10% of the binding data
        df_top10 = process_tf_data_combined(
            tf,
            data,
            formulas,
            filter_top_10=True,
            quantile_threshold=quantile_threshold,
        )

        # Find common features that survived BOTH Lasso regressions
        common_features = set(df_all_data["Feature"]).intersection(df_top10["Feature"])
        common_features_dict[tf] = list(common_features)

    # Return as a dictionary where the response TF is the key, and the value is a
    # list of the surviving features
    return dict(sorted(common_features_dict.items()))


def filter_significant_features(y, X, threshold=0.001):
    """Filter features based on p-value threshold, excluding intercept."""

    # helper method for create_and_fit_combined_filtered_models that does the
    # culling process using a given p-value threshold
    model = sm.OLS(y, X).fit()
    return [
        feature
        for feature, pval in model.pvalues.items()
        if pval < threshold and feature != "Intercept"
    ], model


def create_and_fit_combined_filtered_models(
    common_features_dict, data, full_data_iterations=1, top10_data_iterations=1
):
    """
    Combine filtering phases on the full dataset and the top 10% by LRB_{tf} data.

    Parameters:
    common_features_dict (dict): Dict w/ TFs as keys and list of common features as
    values.
    data (DataFrame): The original DataFrame.
    full_data_iterations (int, optional): Number of filtering iterations on the full
    dataset.
    top10_data_iterations (int, optional): Number of filtering iterations on the
    top 10% data.

    Returns:
    dict: A dictionary where each key is a TF and the value is the final fitted model
    or None if no model survived.

    """
    final_fitted_models: dict[str, Any | None] = {}

    for tf, features in common_features_dict.items():
        current_features = features.copy()
        survived_all_iterations = True
        lrb_column = f"LRB_{tf}"

        # Part 1: Culling features on the entire dataset
        if lrb_column not in current_features:
            current_features.append(lrb_column)

        for _ in range(full_data_iterations):
            formula = f"LRR_{tf} ~ " + " + ".join(current_features)
            y, X = dmatrices(formula, data, return_type="dataframe")
            current_features, model = filter_significant_features(y, X, 0.001)

            if not current_features:
                survived_all_iterations = False
                break

        if not survived_all_iterations:
            final_fitted_models[tf] = None
            continue

        # Part 2: Culling features only based on the top10% by binding data
        top_10_data = data[data[lrb_column] >= data[lrb_column].quantile(0.90)]
        for _ in range(top10_data_iterations):
            formula = f"LRR_{tf} ~ " + " + ".join(current_features)
            y, X = dmatrices(formula, top_10_data, return_type="dataframe")
            current_features, model = filter_significant_features(y, X, threshold=0.01)

            if not current_features:
                survived_all_iterations = False
                break

        # Return the remaining terms
        if survived_all_iterations:
            final_formula = f"LRR_{tf} ~ " + " + ".join(current_features)
            y_final, X_final = dmatrices(
                final_formula, top_10_data, return_type="dataframe"
            )
            final_model = sm.OLS(y_final, X_final).fit()
            final_fitted_models[tf] = final_model
        else:
            final_fitted_models[tf] = None

    return final_fitted_models


def get_existing_formulas_and_tfs(model_dict):
    """
    Extract formulas from existing models in the dictionary and collect TF names from
    LRR_{tf}.

    Parameters:
    model_dict (dict): Dictionary where keys are TFs and values are statsmodels models
    or None.

    Returns:
    tuple: A tuple containing:
        - list: A list of formulas (only for existing models).
        - list: A list of TF names derived from the LRR_{tf} response variables.

    """
    formulas = []
    tf_list = []

    for tf, model in model_dict.items():
        if model is not None:
            response_var = model.model.endog_names
            predictors = [
                pred for pred in model.model.exog_names if pred != "Intercept"
            ]

            # Reconstruct formula as a string
            formula = f"{response_var} ~ " + " + ".join(predictors)
            formulas.append(formula)

            # Extract TF name from response variable
            if response_var.startswith("LRR_"):
                tf_name = response_var.split("LRR_")[1]
                tf_list.append(tf_name)

    return formulas, tf_list


def evaluate_model(formula, train_genes, test_genes, data, response_variable):
    """
    Fits a model using the specified formula and evaluates it on test data.

    Parameters:
    - formula: Patsy formula for the model.
    - train_genes: Genes in the training set.
    - test_genes: Genes in the test set.
    - data: Full DataFrame with predictors and response variable.

    Returns:
    - test_r2: R-squared on the test set.

    """
    # Filter data for training and test sets
    train_data = data[data["target_locus_tag"].isin(train_genes)]
    test_data = data[data["target_locus_tag"].isin(test_genes)]

    # Fit the model on the training set
    y_train, X_train = patsy.dmatrices(formula, train_data, return_type="dataframe")
    model = sm.OLS(y_train, X_train).fit()

    # Predict on the test set and calculate R-squared
    y_test, X_test = patsy.dmatrices(formula, test_data, return_type="dataframe")
    y_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)

    return test_r2


def get_cross_validation_folds_from_dataframe(df, n_splits, tf):
    """
    Generates cross-validation folds while maintaining class distribution.

    Parameters:
    - df: DataFrame containing "target_locus_tag" and "classification".
    - n_splits: Number of folds for cross-validation.
    - tf: The transcription factor of interest.

    Returns:
    - folds: List of (X_train, X_test) for each fold.

    """
    X = df["target_locus_tag"].values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return [
        (X[train_idx].tolist(), X[test_idx].tolist())
        for train_idx, test_idx in skf.split(X, df["classification"])
    ]


def classify_genes(df, tf):
    """
    Classify genes based on binding and perturbation data rankings.

    Parameters:
    - df: DataFrame with data for classification.
    - tf: Transcription factor of interest.

    Returns:
    - classification_df: DataFrame with gene classifications.

    """
    binding_col, perturbation_col = f"LRB_{tf}", f"LRR_{tf}"
    required_columns = {"target_locus_tag", binding_col, perturbation_col}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    classification_df = df[["target_locus_tag", binding_col, perturbation_col]].copy()
    classification_df["binding_rank"] = classification_df[binding_col].rank(
        method="min", ascending=False
    )
    classification_df["perturbation_rank"] = classification_df[perturbation_col].rank(
        method="min", ascending=False
    )

    # Binning ranks and combining classifications
    bins = [0, 8, 64, 512, np.inf]
    classification_df["binding_bin"] = pd.cut(
        classification_df["binding_rank"], bins=bins, labels=[1, 2, 3, 4]
    ).astype(int)
    classification_df["perturbation_bin"] = pd.cut(
        classification_df["perturbation_rank"], bins=bins, labels=[1, 2, 3, 4]
    ).astype(int)
    classification_df["classification"] = (
        classification_df["binding_bin"] - 1
    ) * 4 + classification_df["perturbation_bin"]

    return classification_df.sort_values("classification")


def create_max_LRB_columns(df):
    """
    For each column of the form 'LRB_{tf}' in the DataFrame, create a new column
    'max_LRB_{tf}' that contains the maximum value from all other 'LRB_{other_tf}'
    columns for each row.

    Parameters:
    - df: pandas DataFrame containing columns of the form 'LRB_{tf}'.

    Returns:
    - df: pandas DataFrame with additional 'max_LRB_{tf}' columns.

    """
    lrb_columns = [col for col in df.columns if col.startswith("LRB_")]
    max_columns = {}

    for col in lrb_columns:
        # Calculate max values from all other LRB columns except itself
        other_lrb_columns = [c for c in lrb_columns if c != col]
        max_columns[f"max_{col}"] = df[other_lrb_columns].max(axis=1)

    # Create a new DataFrame from the max_columns dictionary
    max_df = pd.DataFrame(max_columns)

    # Concatenate the new columns with the original DataFrame
    df = pd.concat([df, max_df], axis=1)

    return df


def custom_wrapper_cross_validation(df, n_splits, tf, current_formula):
    """
    Performs cross-validation on different variants of the model formula.

    Parameters:
    - df: DataFrame containing data for models.
    - n_splits: Number of folds for cross-validation.
    - tf: Transcription factor of interest.
    - current_formula: The initial formula for model refinement.

    Returns:
    - results_df: DataFrame containing average R-squared values, formula type, and
    formula.

    """
    formulas_to_test = {"Current": current_formula}
    predictors = current_formula.split(" ~ ")[1]

    # Extract and modify interaction terms
    interaction_terms = [
        term for term in predictors.split(" + ") if term.startswith(f"LRB_{tf}:")
    ]
    for term in interaction_terms:
        main_effect = term.split(":")[1]
        formulas_to_test[f"Replace_{main_effect}"] = current_formula.replace(
            term, main_effect
        )
        formulas_to_test[f"Add_{main_effect}"] = f"{current_formula} + {main_effect}"

    # Generate cross-validation folds
    cv_data = classify_genes(df, tf)
    folds = get_cross_validation_folds_from_dataframe(cv_data, n_splits, tf)

    # Perform cross-validation for each formula
    results = []
    for formula_name, formula in formulas_to_test.items():
        r2_list = [
            evaluate_model(formula, train, test, df, tf) for train, test in folds
        ]
        avg_r2 = sum(r2_list) / len(r2_list)
        results.append(
            {
                "TF": tf,
                "Formula_Type": formula_name,
                "Avg_R_squared": avg_r2,
                "Formula": formula,
            }
        )

    return pd.DataFrame(results)


def iterative_model_selection(df, n_splits, tf, final_formulas):
    """
    Iteratively refines the model by evaluating each interaction term and returns a
    DataFrame with the response TF, each feature in the final formula, and their
    coefficients sorted by the magnitude of the coefficient (excluding the intercept).

    Parameters:
    - df: pandas DataFrame containing the data for the models.
    - n_splits: Number of folds for cross-validation.
    - tf: The transcription factor of interest.
    - final_formulas: A list of final formulas output by complex models to get the
    multivariate formula.

    Returns:
    - result_df: DataFrame containing the response TF, each feature in the final
    formula, and the coefficient of that feature, sorted by coefficient magnitude.

    """
    # Find the original multivariate formula from the list
    multivariate_formula = next(
        (s for s in final_formulas if s.startswith(f"LRR_{tf}")), None
    )
    if multivariate_formula is None:
        raise ValueError(f"No multivariate formula found for {tf}.")

    logger.info(
        f"Starting iterative model selection w/ og formula: {multivariate_formula}"
    )

    # Split the formula into response and terms
    response_variable, terms = multivariate_formula.split(" ~ ")
    terms = terms.split(" + ")

    # Calculate the average R-squared for the original formula
    initial_results_df = custom_wrapper_cross_validation(
        df, n_splits, tf, multivariate_formula
    )
    original_avg_r2 = initial_results_df.loc[
        initial_results_df["Formula_Type"] == "Current", "Avg_R_squared"
    ].mean()

    # Extract interaction terms from the original formula
    interaction_terms = [term for term in terms if ":" in term]

    # We make a dict. of necessary modifications so that we can keep track of
    # everything that needs to be updated at the end
    modifications = {}
    best_avg_r2 = original_avg_r2

    for interaction_term in interaction_terms:
        main_effect = interaction_term.split(":")[1]

        iteration_results_df = custom_wrapper_cross_validation(
            df, n_splits, tf, multivariate_formula
        )

        # Filter for the specific formulas from the CV output based on the current
        # interaction term
        relevant_formulas = iteration_results_df[
            iteration_results_df["Formula_Type"].isin(
                ["Current", f"Replace_{main_effect}", f"Add_{main_effect}"]
            )
        ]

        best_model = relevant_formulas.loc[relevant_formulas["Avg_R_squared"].idxmax()]
        if best_model["Avg_R_squared"] > best_avg_r2:
            best_avg_r2 = best_model["Avg_R_squared"]

            # Keep track of whether an addition or substitution is needed
            if best_model["Formula_Type"] == f"Replace_{main_effect}":
                modifications[interaction_term] = ("replace", main_effect)
            elif best_model["Formula_Type"] == f"Add_{main_effect}":
                modifications[interaction_term] = ("add", main_effect)

            logger.info(
                f"Selected modification for '{interaction_term}': "
                f"{best_model['Formula_Type']} with Avg R-squared = {best_avg_r2}"
            )

    # Make all substitutions/additions at the same time to update the formula
    final_terms = terms.copy()

    for interaction_term, (action, main_effect) in modifications.items():
        if action == "replace":
            final_terms = [
                term if term != interaction_term else main_effect
                for term in final_terms
            ]
        elif action == "add":
            if main_effect not in final_terms:
                final_terms.append(main_effect)

    # Obtain the final formula and fit it to the data
    final_formula = f"{response_variable} ~ {' + '.join(final_terms)}"
    logger.info(f"Final formula after all modifications: {final_formula}")
    y, X = patsy.dmatrices(final_formula, data=df, return_type="dataframe")
    model = sm.OLS(y, X).fit()
    coefficients = model.params.drop(
        "Intercept", errors="ignore"
    )  # Exclude the intercept

    # Return results sorted by coefficient magnitude
    result_df = (
        pd.DataFrame(
            {
                "Response TF": tf,
                "Feature": coefficients.index,
                "Coefficient": coefficients.values,
            }
        )
        .sort_values(by="Coefficient", key=abs, ascending=False)
        .reset_index(drop=True)
    )

    return result_df


def add_main_effects(final_models, data):
    final_formulas, final_tf_list = get_existing_formulas_and_tfs(final_models)

    combined_results_final_df = pd.DataFrame()
    # Loop through each tf and collect results
    for tf in final_tf_list:
        results_df = iterative_model_selection(data, 4, tf, final_formulas)
        combined_results_final_df = pd.concat(
            [combined_results_final_df, results_df], ignore_index=True
        )
    return combined_results_final_df
