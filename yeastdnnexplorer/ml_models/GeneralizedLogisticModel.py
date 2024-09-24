import numpy as np


class GeneralizedLogisticModel:
    def __init__(self, A, D, B, C, E=None):
        """
        Generalized logistic function with an interactor term.

        This function calculates the value of a generalized logistic model, where the
        slope parameter `B` is modified by an interaction term `Y_interactor`.
        The model takes the form:

        .. math::
            Y(X) = \frac{A - D}{1 + e^{-(B + E(Y_{\text{interactor}} - C))(X - C)}} + D

        Where:

        - :math:`A` is the upper asymptote
            (the maximum value as :math:`X \to \infty`).
        - :math:`D` is the lower asymptote
            (the minimum value as :math:`X \to -\infty`).
        - :math:`B` controls the slope (steepness) of the curve
            at the inflection point.
        - :math:`C` is the inflection point, where the rate of
            change of the curve is highest.
        - :math:`E` is the interaction coefficient that modifies
            the effect of the interactor `Y_interactor`.
        - :math:`Y_{\text{interactor}}` is the interactor
            variable that influences the slope.

        Parameters
        ----------
        X : float or ndarray
            The independent variable (e.g., time, concentration).

        A : float
            Upper asymptote (maximum value of the curve as :math:`X \to \infty`).

        D : float
            Lower asymptote (minimum value of the curve as :math:`X \to -\infty`).

        B : float
            Slope parameter at the inflection point (steepness of the curve).

        C : float
            Inflection point, where the curve changes its direction most rapidly.

        E : float
            Interaction coefficient, representing the strength of
            the interactor's influence on the slope.

        Y_interactor : float or ndarray
            Interactor variable that modifies the slope parameter `B`.

        Returns
        -------
        Y : float or ndarray
            The value of the logistic function evaluated at `X`.

        Example
        -------
        >>> X = 50
        >>> A = 100
        >>> D = 10
        >>> B = 0.1
        >>> C = 50
        >>> E = 0.5
        >>> Y_interactor = 20
        >>> generalized_logistic_with_interactor(X, A, D, B, C, E, Y_interactor)
        55.8823
        """
        self.A = A
        self.D = D
        self.B = B
        self.C = C
        self.E = E  # Interaction coefficient for the interactor

    def _sigmoid(self, X, Y_interactor=None):
        """
        Calculate the logistic function for a given input X.
        If an interactor (Y_interactor) is provided, it will modify the slope.
        """
        if Y_interactor is not None and self.E is not None:
            # Modify the slope using the interactor and its effect
            modified_slope = self.B + self.E * (Y_interactor - self.C)
        else:
            # No interactor; use the base slope B
            modified_slope = self.B

        # Apply the logistic formula
        return (self.A - self.D) / (1 + np.exp(-modified_slope * (X - self.C))) + self.D

    def predict(self, X, Y_interactor=None):
        """
        Make predictions using the logistic model. Include interactor if provided.
        """
        return self._sigmoid(X, Y_interactor)

    def fit(self, X, Y, Y_interactor=None):
        """
        (Optional) Fit the model to data, adjusting parameters A, D, B, C, E
        using some optimization algorithm like gradient descent or curve fitting.
        """
        # Implementation of fitting logic if needed
        pass
