import logging
import numbers
from typing import Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats
from scipy.optimize import curve_fit
from tabulate import tabulate

logger = logging.getLogger("general")


def sigmoid(
    X: NDArray[np.float_],  # X should be a 2D array of floats
    upper_asymptote: float,
    lower_asymptote: float,
    inflection_point: Union[
        NDArray[np.float_], float
    ],  # inflection point can be a 1D array or a scalar
    B: Union[NDArray[np.float_], float],  # B can also be a 1D array or scalar
) -> NDArray[np.float_]:
    """
    Generalized sigmoid function for multiple variables.

    This function calculates a generalized sigmoid where the slope parameters
    `B_i` are applied to each variable, with the inflection point shifting each
    variable independently.

    $$
    Y(X) = \\frac{upper\\_asymptote - lower\\_asymptote}
    {1 + \\exp\\left( -\\sum_{i=1}^{n} B_i (X_i - inflection\\_point_i) \\right)}
    + lower\\_asymptote
    $$

    :param X: Input data matrix (2D array).
    :param upper_asymptote: Upper asymptote (maximum value of the curve).
    :param lower_asymptote: Lower asymptote (minimum value of the curve).
    :param inflection_point: Inflection point of the sigmoid (1D array or scalar).
    :param B: Slope coefficients for each variable (1D array or scalar).

    :return: The value of the logistic function at X.
    :rtype: np.ndarray
    """
    if lower_asymptote >= upper_asymptote:
        raise ValueError("Lower asymptote must be less than the upper asymptote.")

    # Convert to numpy arrays if they aren't already
    X = np.asarray(X)
    inflection_point = np.atleast_1d(inflection_point)
    B = np.atleast_1d(B)

    # Ensure B has the same length as the number of columns in X
    if len(B) != X.shape[1]:
        raise ValueError(f"Expected {X.shape[1]} coefficients, but got {len(B)}.")

    # Ensure the inflection_point has the same dimension as X
    if inflection_point.shape != (X.shape[1],):
        raise ValueError(
            "Inflection point must have the same number of elements as columns in X."
        )

    # Calculate the linear combination of adjusted X and B
    linear_combination = np.dot(X - inflection_point, B)

    # Apply the sigmoid function
    result = (upper_asymptote - lower_asymptote) / (
        1 + np.exp(-linear_combination)
    ) + lower_asymptote

    return result


class GeneralizedLogisticModel:
    def __init__(self):
        """
        Generalized logistic function with an interactor term.
        """
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._upper_asymptote: float | None = None
        self._lower_asymptote: float | None = None
        self._inflection_point: np.ndarray | None = None
        self._coefficients: np.ndarray | None = None
        self._cov: np.ndarray | None = None
        self._residuals: np.ndarray | None = None

    @property
    def X(self) -> np.ndarray | None:
        """
        Set the predictor variables for the model.

        :param value: The input data matrix. Must be two dimensional even if there is
            only one predictor.
        :type value: np.ndarray

        :return: The input data matrix.
        :rtype: np.ndarray

        :raises: TypeError if X is not a NumPy array.
        :raises: ValueError if X is not 2D.
        :raises: ValueError if the number of columns in X does not match the length of
            the inflection point or coefficients.
        """
        return self._X

    @X.setter
    def X(self, value: np.ndarray) -> None:
        if not isinstance(value, np.ndarray):
            raise TypeError("X must be a NumPy array.")

        # Ensure that X is 2D
        if value.ndim != 2:
            raise ValueError("X must be a 2-dimensional array.")

        # Check compatibility with inflection_point
        if (
            self.inflection_point is not None
            and len(self.inflection_point) != value.shape[1]
        ):
            raise ValueError(
                "Inflection point must have the same "
                "number of elements as columns in X."
            )

        # Check compatibility with coefficients
        if self.coefficients is not None and len(self.coefficients) != value.shape[1]:
            raise ValueError(
                "Coefficients must have the same number of elements as columns in X."
            )

        self._X = value

    @property
    def y(self) -> np.ndarray | None:
        """
        Set the response variable for the model.

        :param value: The observed output data.
        :type value: np.ndarray

        :return: The observed output data.
        :rtype: np.ndarray

        :raises: TypeError if y is not a NumPy array or a list.
        :raises: ValueError if the number of rows in y
            does not match the number of rows in X.
        """
        return self._y

    @y.setter
    def y(self, value: np.ndarray) -> None:
        if not isinstance(value, (np.ndarray, list)):
            raise TypeError("y must be a NumPy array or a list.")

        value = np.asarray(value)

        if self.X is not None and value.shape[0] != self.X.shape[0]:
            raise ValueError("y must have the same number of rows as X.")

        self._y = value

    @property
    def upper_asymptote(self) -> float | None:
        """
        Set the upper asymptote for the model. This parameter can be inferred by
            `fit()`

        :param value: The upper asymptote of the sigmoid function.
        :type value: float

        :return: The upper asymptote of the sigmoid function.
        :rtype: float

        :raises: TypeError if the upper asymptote is not a real number.
        :raises: ValueError if the upper asymptote is less than the lower asymptote.
        """
        return self._upper_asymptote

    @upper_asymptote.setter
    def upper_asymptote(self, value: float) -> None:
        if not isinstance(value, numbers.Real):
            raise TypeError("Upper asymptote must be a real number.")

        if self.lower_asymptote is not None and value <= self.lower_asymptote:
            raise ValueError(
                "Upper asymptote must be greater than the lower asymptote."
            )

        self._upper_asymptote = value

    @property
    def lower_asymptote(self) -> float | None:
        """
        The lower asymptote of the sigmoid function. This parameter can be inferred by
            `fit()`

        :return: The lower asymptote of the sigmoid function.
        :rtype: float

        :raises: TypeError if the lower asymptote is not a real number.
        :raises: ValueError if the lower asymptote is greater than the upper asymptote.
        """
        return self._lower_asymptote

    @lower_asymptote.setter
    def lower_asymptote(self, value: float) -> None:
        if not isinstance(value, numbers.Real):
            raise TypeError("Lower asymptote must be a real number.")

        if self.upper_asymptote is not None and value >= self.upper_asymptote:
            raise ValueError("Lower asymptote must be less than the upper asymptote.")

        self._lower_asymptote = value

    @property
    def inflection_point(self) -> np.ndarray | None:
        """
        Set the inflection point for the model. This parameter can be inferred by
            `fit()`

        :param value: The inflection point of the sigmoid function.
        :type value: np.ndarray

        :return: The inflection point of the sigmoid function.
        :rtype: np.ndarray

        :raises: TypeError if the inflection point is not a NumPy array or a list.
        :raises: ValueError if the length of the inflection point does not match the
            number of columns in X or the number of coefficients.
        """
        return self._inflection_point

    @inflection_point.setter
    def inflection_point(self, value: Sequence[float] | np.ndarray) -> None:
        value = np.asarray(value)  # Convert to NumPy array if it's not already

        # Ensure compatibility with X
        if self.X is not None and len(value) != self.X.shape[1]:
            raise ValueError(
                "Inflection point must have the same "
                "number of elements as columns in X."
            )

        # Ensure compatibility with coefficients
        if self.coefficients is not None and len(value) != len(self.coefficients):
            raise ValueError(
                "Inflection point must have the same "
                "number of elements as coefficients."
            )

        self._inflection_point = value

    @property
    def coefficients(self) -> np.ndarray | None:
        """
        Set the coefficients for the model. This parameter can be inferred by `fit()`

        :param value: The coefficients of the sigmoid function.
        :type value: np.ndarray

        :return: The coefficients of the sigmoid function.
        :rtype: np.ndarray

        :raises: TypeError if the coefficients are not a NumPy array or a list.
        :raises: ValueError if the length of the coefficients does not match the number
            of columns in X or the number of inflection points.
        """
        return self._coefficients

    @coefficients.setter
    def coefficients(self, value: Sequence[float] | np.ndarray) -> None:
        # validate type
        if not isinstance(value, (np.ndarray, list)):
            raise TypeError("Coefficients must be a NumPy array or a list.")

        value = np.asarray(value)

        if self.X is not None and len(value) != self.X.shape[1]:
            raise ValueError(
                "Coefficients must have the same number of elements as columns in X."
            )

        if self.inflection_point is not None and len(value) != len(
            self.inflection_point
        ):
            raise ValueError(
                "Coefficients must have the same number "
                "of elements as inflection point."
            )

        self._coefficients = value

    @property
    def cov(self) -> np.ndarray | None:
        """
        The covariance matrix of the model parameters. This parameter can be inferred by
            `fit()`

        :return: The covariance matrix of the model parameters.
        :rtype: np.ndarray

        :raises: TypeError if the covariance matrix is not a NumPy array.
        """
        return self._cov

    @cov.setter
    def cov(self, value: np.ndarray) -> None:
        if not isinstance(value, np.ndarray):
            raise TypeError("cov must be a NumPy array.")
        self._cov = value

    @property
    def residuals(self) -> np.ndarray | None:
        """
        The residuals of the model. This parameter can be inferred by `fit()`

        :return: The residuals of the model.
        :rtype: np.ndarray

        :raises: TypeError if the residuals are not a NumPy array.
        """
        return self._residuals

    @residuals.setter
    def residuals(self, value: np.ndarray) -> None:
        if not isinstance(value, np.ndarray):
            raise TypeError("residuals must be a NumPy array.")
        self._residuals = value

    def __call__(self, X: ArrayLike):
        """
        This passes the arguments to the predict method.

        :raises: ValueError if the arguments required by predict() are not provided.

        """
        return self.predict(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the generalized logistic model.

        :param X: Input data matrix
        :type X: np.ndarray

        :return: Predictions based on the learned model parameters
        :rtype: np.ndarray

        :raises: ValueError if the model has not been fitted.
        """
        if self.upper_asymptote is None or self.lower_asymptote is None:
            raise ValueError("Model must be fitted before making predictions.")

        assert self.inflection_point is not None, "Inflection point is not available."
        assert self.coefficients is not None, "Coefficients are not available."

        return sigmoid(
            X,
            self.upper_asymptote,
            self.lower_asymptote,
            self.inflection_point,
            self.coefficients,
        )

    def model(self, y: np.ndarray, X: np.ndarray) -> None:
        """
        Set the predictor and response variables for the model.

        :param X: The input data matrix. Must be two dimensional even if there is
            only one predictor.
        :type X: np.ndarray
        :param y: The observed output data.
        :type y: np.ndarray
        """
        print(f"Input y shape: {y.shape}")  # Check the shape of X before setting
        self.y = y
        print(f"Set X shape: {X.shape}")  # Verify after setting
        self.X = X  # This should set the 2D array correctly

    def fit(self, **kwargs) -> None:
        """Fit the model to the data
        :param kwargs: Additional keyword arguments. These include:

            - **upper_asymptote**: Initial guess for the upper asymptote.
                Defaults to 1.0.
            - **lower_asymptote**: Initial guess for the lower asymptote.
                Defaults to 0.0.
            - **inflection_point**: Initial guess for the inflection point.
            - **initial_coefficients**: Initial guess for the coefficients.
            - Any other kwargs are passed on to `scipy.optimize.curve_fit`.

        """

        assert (
            self.X is not None
        ), "Input data matrix X is not available. Set via `model()`"
        assert self.y is not None, "Output data y is not available. Set via `model()`"
        initial_upper_asymptote = kwargs.pop("upper_asymptote", 1.0)
        initial_lower_asymptote = kwargs.pop("lower_asymptote", 0.0)
        initial_inflection_point = kwargs.pop(
            "inflection_point", np.mean(self.X, axis=0)
        )
        initial_coefficients = kwargs.pop("coefficients", np.ones(self.X.shape[1]))

        # Flatten the parameters into a single array for curve_fit
        initial_params = (
            [initial_upper_asymptote, initial_lower_asymptote]
            + initial_inflection_point.tolist()
            + initial_coefficients.tolist()
        )

        # Define the model for curve_fit to optimize
        def model_to_fit(X, upper_asymptote, lower_asymptote, *params):
            n = X.shape[1]
            inflection_point = np.array(params[:n])
            coefficients = np.array(params[n:])
            return sigmoid(
                X, upper_asymptote, lower_asymptote, inflection_point, coefficients
            )

        # Use curve_fit to optimize the model parameters
        logger.info("Fitting the generalized logistic model to the data.")
        popt, cov, infodict, mesg, ier = curve_fit(
            f=model_to_fit,
            xdata=self.X,
            ydata=self.y,
            p0=initial_params,
            full_output=True,
            **kwargs,
        )

        # Extract the optimized parameters from popt
        self.upper_asymptote = popt[0]
        self.lower_asymptote = popt[1]
        self.inflection_point = np.array(popt[2 : self.X.shape[1] + 2])
        self.coefficients = np.array(popt[self.X.shape[1] + 2 :])
        self.cov = cov
        self.residuals = infodict.get("fvec")

    def summary(self) -> None:
        """
        Generate a summary of the model and diagnostic statistics.
        """
        if self.X is None or self.y is None or self.coefficients is None:
            raise ValueError("Model must be fitted before generating a summary.")

        # Calculate the number of parameters and degrees of freedom
        assert self.inflection_point is not None, "Inflection point is not available."
        n_params = len(self.coefficients) + len(self.inflection_point) + 2
        n_obs = len(self.y)
        df_residual = n_obs - n_params  # Degrees of freedom for residuals

        # Calculate residual sum of squares (RSS) and total sum of squares (TSS)
        assert self.residuals is not None, "Residuals are not available."
        rss = np.sum(self.residuals**2)  # Residual Sum of Squares
        tss = np.sum((self.y - np.mean(self.y)) ** 2)  # Total Sum of Squares

        # Calculate R-squared and Adjusted R-squared
        r_squared = 1 - (rss / tss)
        adj_r_squared = 1 - (1 - r_squared) * ((n_obs - 1) / df_residual)

        # Residual Standard Error
        residual_std_error = np.sqrt(rss / df_residual)

        # Extract standard errors from the covariance matrix
        assert self.cov is not None, "Covariance matrix is not available."
        std_errors = np.sqrt(np.diag(self.cov))

        # t-values and p-values for the coefficients
        # (including asymptotes and inflection points)
        estimates = np.concatenate(
            [
                [self.upper_asymptote, self.lower_asymptote],
                self.inflection_point,
                self.coefficients,
            ]
        )
        t_values = estimates / std_errors
        p_values = 2 * (
            1 - stats.t.cdf(np.abs(t_values), df_residual)
        )  # Two-tailed p-values

        # F-statistic
        ms_model = (tss - rss) / (n_params - 1)
        ms_residual = rss / df_residual
        f_statistic = ms_model / ms_residual
        p_f_statistic = 1 - stats.f.cdf(f_statistic, n_params - 1, df_residual)

        # Prepare the summary table
        param_names = (
            ["upper_asymptote", "lower_asymptote"]
            + [f"inflection_point_{i}" for i in range(len(self.inflection_point))]
            + [f"coef_{i}" for i in range(len(self.coefficients))]
        )

        table_data = [
            [param, f"{est:12.4f}", f"{se:12.4f}", f"{t_val:12.4f}", f"{p_val:12.4g}"]
            for param, est, se, t_val, p_val in zip(
                param_names, estimates, std_errors, t_values, p_values
            )
        ]

        # Print the summary
        print("\nGeneralized Logistic Model Summary\n")

        print(
            tabulate(
                table_data,
                headers=["", "Estimate", "Std. Error", "t value", "Pr(>|t|)"],
                tablefmt="pipe",
            )
        )

        print(
            f"\nResidual standard error: {residual_std_error:.4f} "
            f"on {df_residual} degrees of freedom"
        )
        print(f"R-squared: {r_squared:.4f}, Adjusted R-squared: {adj_r_squared:.4f}")
        print(
            f"F-statistic: {f_statistic:.4f} on {n_params - 1} "
            f"and {df_residual} DF, p-value: {p_f_statistic:.4g}"
        )

    def plot(self, plots_to_display: list[int] = [1, 2, 3, 4]):
        """
        Diagnostic plots for the generalized logistic model. Similar to R's `lm()`
        diagnostic plots, this function displays:

        - 1: Residuals vs Fitted
        - 2: Normal Q-Q plot
        - 3: Scale-Location plot
        - 4: Residuals vs Leverage

        The user can specify a subset of these plots by providing a list of integers.
        For example, passing [1, 3] will show the Residuals vs Fitted
        and Scale-Location plots.

        :param plots_to_display: A list of integers (1 to 4) indicating
            which plots to show.
        :type plots_to_display: list[int]
        """

        if self.X is None or self.y is None or self.residuals is None:
            raise ValueError("Model must be fitted before plotting diagnostics.")

        # Calculate fitted values
        fitted_values = self.y - self.residuals

        # Create subplots grid based on number of plots to display
        n_plots = len(plots_to_display)
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))

        if n_plots == 1:
            axes = [axes]  # Ensure axes is always iterable

        for i, plot_num in enumerate(plots_to_display):
            if plot_num == 1:
                # 1. Residuals vs Fitted
                axes[i].scatter(fitted_values, self.residuals, alpha=0.5)
                axes[i].axhline(0, color="gray", linestyle="--")
                axes[i].set_xlabel("Fitted values")
                axes[i].set_ylabel("Residuals")
                axes[i].set_title("Residuals vs Fitted")

            elif plot_num == 2:
                # 2. Normal Q-Q Plot
                stats.probplot(self.residuals, dist="norm", plot=axes[i])
                axes[i].set_title("Normal Q-Q")

            elif plot_num == 3:
                # 3. Scale-Location (Spread-Location plot)
                sqrt_residuals = np.sqrt(np.abs(self.residuals))
                axes[i].scatter(fitted_values, sqrt_residuals, alpha=0.5)
                axes[i].set_xlabel("Fitted values")
                axes[i].set_ylabel("Sqrt(|Residuals|)")
                axes[i].set_title("Scale-Location")

            elif plot_num == 4:
                # 4. Residuals vs Leverage
                leverage = np.diag(
                    np.dot(
                        self.X, np.linalg.pinv(np.dot(self.X.T, self.X)).dot(self.X.T)
                    )
                )  # Approx leverage
                axes[i].scatter(leverage, self.residuals, alpha=0.5)
                axes[i].set_xlabel("Leverage")
                axes[i].set_ylabel("Residuals")
                axes[i].set_title("Residuals vs Leverage")
                axes[i].axhline(0, color="gray", linestyle="--")

        plt.tight_layout()
        plt.show()
