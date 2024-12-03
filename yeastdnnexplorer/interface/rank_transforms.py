import numpy as np
from scipy.stats import rankdata


def shifted_negative_log_ranks(ranks: np.ndarray) -> np.ndarray:
    """
    Transforms ranks to negative log10 values and shifts such that the lowest value is
    0.

    :param ranks: A vector of ranks
    :return np.ndarray: A vector of negative log10 transformed ranks shifted such that
        the lowest value is 0
    :raises ValueError: If the ranks are not numeric.

    """
    if not np.issubdtype(ranks.dtype, np.number):
        raise ValueError("`ranks` must be a numeric")
    max_rank = np.max(ranks)
    log_max_rank = np.log10(max_rank)
    return -1 * np.log10(ranks) + log_max_rank


def stable_rank(pvalue_vector: np.ndarray, enrichment_vector: np.ndarray) -> np.ndarray:
    """
    Ranks data by primary_column, breaking ties based on secondary_column. The expected
    primary and secondary columns are 'pvalue' and 'enrichment', respectively. Then the
    ranks are transformed to negative log10 values and shifted such that the lowest
    value is 0 and the highest value is log10(min_rank).

    :param pvalue_vector: A vector of pvalues
    :param enrichment_vector: A vector of enrichment values corresponding to the pvalues
    :return np.ndarray: A vector of negative log10 transformed ranks shifted such that
        the lowest value is 0 and the highest value is log10(min_rank)
    :raises ValueError: If the primary or secondary column is not numeric.

    """

    # Check if primary and secondary columns are numeric
    if not np.issubdtype(pvalue_vector.dtype, np.number):
        raise ValueError("`primary_vector` must be a numeric")
    if not np.issubdtype(enrichment_vector.dtype, np.number):
        raise ValueError("`secondary_vector` must be a numeric")

    # Step 1: Rank by primary_column
    # note that this will now always be an integer, unlike average which could return
    # decimal values making adding the secondary rank more difficult
    primary_rank = rankdata(pvalue_vector, method="min")

    # Step 2: Identify ties in primary_rank
    unique_ranks = np.unique(primary_rank)

    # Step 3: Adjust ranks within ties using secondary ranking
    adjusted_primary_rank = primary_rank.astype(
        float
    )  # Convert to float for adjustments

    for unique_rank in unique_ranks:
        # Get indices where primary_rank == unique_rank
        tie_indices = np.where(primary_rank == unique_rank)[0]

        if len(tie_indices) > 1:  # Only adjust if there are ties
            # Rank within the tie group by secondary_column
            # (descending if higher is better)
            tie_secondary_values = enrichment_vector[tie_indices]
            secondary_rank_within_ties = rankdata(
                -tie_secondary_values, method="average"
            )

            # Calculate dynamic scale factor to ensure adjustments are < 1. Since the
            # primary_rank is an integer, adding a number less than 1 will not affect
            # rank relative to the other groups.
            max_secondary_rank = np.max(secondary_rank_within_ties)
            scale_factor = (
                0.9 / max_secondary_rank
            )  # Keep scale factor slightly below 1/max rank

            # multiple the secondary_rank_within_ties values by 0.1 and add this value
            # to the adjusted_primary_rank_values. This will rank the tied primary
            # values by the secondary values, but not affect the overall primary rank
            # outside of the tie group
            # think about this scale factor
            adjusted_primary_rank[tie_indices] += (
                secondary_rank_within_ties * scale_factor
            )

    # Step 4: Final rank based on the adjusted primary ranks
    final_ranks = rankdata(adjusted_primary_rank, method="average")

    return final_ranks


def negative_log_transform_by_pvalue_and_enrichment(
    pvalue_vector: np.ndarray, enrichment_vector: np.ndarray
) -> np.ndarray:
    """
    This calls the rank() function and then transforms the ranks to negative log10
    values and shifts to the right such that the lowest value (largest rank,
    least important) is 0.

    :param pvalue_vector: A vector of pvalues
    :param enrichment_vector: A vector of enrichment values corresponding to the pvalues
    :return np.ndarray: A vector of negative log10 transformed ranks shifted such that
        the lowest value is 0 and the highest value is log10(min_rank)
    :raises ValueError: If the primary or secondary column is not numeric.

    """

    final_ranks = stable_rank(pvalue_vector, enrichment_vector)

    return shifted_negative_log_ranks(final_ranks)
