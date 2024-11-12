import pandas as pd
import numpy as np
import pytest
from sklearn.linear_model import LassoCV
from iterative_culling import (
    create_formulas_with_max_lrb, process_tf_data_combined, 
    find_common_features_across_tfs, filter_significant_features, 
    create_and_fit_combined_filtered_models, get_existing_formulas_and_tfs, 
    evaluate_model, get_cross_validation_folds_from_dataframe, classify_genes, 
    create_max_LRB_columns, custom_wrapper_cross_validation, 
    iterative_model_selection, add_main_effects
)


# Sample Data for Tests
@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'target_locus_tag': [f'gene{i}' for i in range(100)],
        'LRB_TF1': np.random.rand(100),
        'LRB_TF2': np.random.rand(100),
        'LRB_TF3': np.random.rand(100),
        'LRR_TF1': np.random.rand(100),
        'LRR_TF2': np.random.rand(100),
        'LRR_TF3': np.random.rand(100)
    })


@pytest.fixture
def tf_list():
    return ['TF1', 'TF2', 'TF3']


@pytest.fixture
def max_formulas(tf_list):
    return create_formulas_with_max_lrb(tf_list)


# Tests for create_formulas_with_max_lrb
def test_create_formulas_with_max_lrb(tf_list):
    formulas = create_formulas_with_max_lrb(tf_list)
    assert len(formulas) == len(tf_list)
    for tf, formula in zip(tf_list, formulas):
        assert f"LRR_{tf}" in formula
        assert f"LRB_{tf}" in formula
        assert f"max_LRB_{tf}" in formula


# Tests for create_max_LRB_columns
def test_create_max_LRB_columns(sample_data, tf_list):
    df_with_max_lrb = create_max_LRB_columns(sample_data.copy())
    for tf in tf_list:
        assert f"max_LRB_{tf}" in df_with_max_lrb.columns


# Tests for find_common_features_across_tfs
def test_find_common_features_across_tfs(tf_list, sample_data, max_formulas):
    common_features = find_common_features_across_tfs(tf_list, sample_data, max_formulas)
    assert isinstance(common_features, dict)
    for tf in tf_list:
        assert tf in common_features


# Tests for process_tf_data_combined
def test_process_tf_data_combined(tf_list, sample_data, max_formulas):
    results_df = process_tf_data_combined("TF1", sample_data, max_formulas, filter_top_10=False)
    assert "Feature" in results_df.columns
    assert "Coefficient" in results_df.columns


# Tests for create_and_fit_combined_filtered_models
def test_create_and_fit_combined_filtered_models(tf_list, sample_data, max_formulas):
    common_features = find_common_features_across_tfs(tf_list, sample_data, max_formulas)
    final_models = create_and_fit_combined_filtered_models(common_features, sample_data)
    assert isinstance(final_models, dict)
    for tf in tf_list:
        assert tf in final_models


# Tests for iterative_model_selection
def test_iterative_model_selection(tf_list, sample_data, max_formulas):
    results_df = iterative_model_selection(sample_data, 3, "TF1", max_formulas)
    assert "Response TF" in results_df.columns
    assert "Feature" in results_df.columns
    assert "Coefficient" in results_df.columns


# Tests for add_main_effects
def test_add_main_effects(tf_list, sample_data, max_formulas):
    common_features = find_common_features_across_tfs(tf_list, sample_data, max_formulas)
    final_models = create_and_fit_combined_filtered_models(common_features, sample_data)
    final_results = add_main_effects(final_models, sample_data)
    assert "Response TF" in final_results.columns
    assert "Feature" in final_results.columns
    assert "Coefficient" in final_results.columns
