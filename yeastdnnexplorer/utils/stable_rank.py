from typing import Union

import numpy as np
from scipy.stats import rankdata


def stable_rank(
    col1: np.ndarray,
    col2: Union[np.ndarray, None] = None,
    col1_ascending: bool = True,
    col2_ascending: bool = True,
    method="average",
) -> np.ndarray:
    """ """
    # Validate inputs
    if not np.issubdtype(col1.dtype, np.number):
        raise ValueError("`col1` must be numeric")
    if col2 is not None and not np.issubdtype(col2.dtype, np.number):
        raise ValueError("`col2` must be numeric if provided")

    # Rank `col1`
    col1_sorted = col1 if col1_ascending else -col1
    primary_rank = rankdata(col1_sorted, method="min")

    if col2 is None:
        # If `col2` is not provided, return ranks based only on `col1`
        return primary_rank

    # Handle ties using `col2`
    adjusted_primary_rank = primary_rank.astype(float)
    unique_ranks = np.unique(primary_rank)

    for unique_rank in unique_ranks:
        tie_indices = np.where(primary_rank == unique_rank)[0]

        if len(tie_indices) > 1:  # Adjust only in case of ties
            col2_sorted = col2[tie_indices] if col2_ascending else -col2[tie_indices]
            secondary_rank_within_ties = rankdata(col2_sorted, method=method)

            # Dynamically scale secondary ranks to prevent overlaps
            max_secondary_rank = np.max(secondary_rank_within_ties)
            scale_factor = 0.9 / max_secondary_rank

            adjusted_primary_rank[tie_indices] += (
                secondary_rank_within_ties * scale_factor
            )

    # Final rank
    final_ranks = rankdata(adjusted_primary_rank, method=method)
    return final_ranks
