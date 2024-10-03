import numpy as np
from numpy.typing import NDArray


def sigmoid(
    X: NDArray[np.float_],  # X should be a 2D array of floats
    left_asymptote: float,
    right_asymptote: float,
    B: NDArray[np.float_] | float,  # B can also be a 1D array or scalar
) -> NDArray[np.float_]:
    """
    Generalized sigmoid function for multiple variables.

    This function calculates a generalized sigmoid where the slope parameters
    `B_i` are applied to each variable, with the inflection point shifting each
    variable independently.

    $$
    Y(X) = \\frac{upper\\_asymptote - lower\\_asymptote}
    {1 + \\exp\\left( -\\sum_{i=1}^{n} B_i * X_i \\right)}
    + lower\\_asymptote
    $$

    :param X: Input data matrix (2D array). The first dimension must be a constant
        vector of ones.
    :param right_asymptote: Upper asymptote (maximum value of the curve).
    :param left_asymptote: Lower asymptote (minimum value of the curve).
    :param B: Slope coefficients for each variable (1D array or scalar).

    :return: The value of the logistic function at X.
    :rtype: np.ndarray

    """
    # Convert to numpy arrays if they aren't already
    X = np.asarray(X)

    # ensure that X[0] is entirely equal to 1
    if not np.all(X[:, 0] == 1):
        raise ValueError("The first column of X must be a constant vector of ones.")

    B = np.atleast_1d(B)

    # Ensure B has the same length as the number of columns in X
    if len(B) != X.shape[1]:
        raise ValueError(f"Expected {X.shape[1]} coefficients, but got {len(B)}.")

    # Calculate the linear combination of adjusted X and B
    linear_combination = np.dot(X, B)

    # Apply the sigmoid function
    numerator = right_asymptote - left_asymptote

    # see https://en.wikipedia.org/wiki/Generalised_logistic_function
    C = 1
    denominator = C + np.exp(-linear_combination)
    result = (numerator / denominator) + left_asymptote

    return result
