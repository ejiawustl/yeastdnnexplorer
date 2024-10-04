import numpy as np
import pandas as pd
import pytest

from yeastdnnexplorer.ml_models.GeneralizedLogisticModel import GeneralizedLogisticModel
from yeastdnnexplorer.utils.InteractorDiagnosticPlot import InteractorDiagnosticPlot
from yeastdnnexplorer.utils.sigmoid import sigmoid


@pytest.fixture
def setup_glm_two_variable():
    glm = GeneralizedLogisticModel()

    # Generate sample data
    np.random.seed(42)
    X1 = np.linspace(-10, 10, 100)
    X2 = np.linspace(-10, 10, 100)
    X0 = np.ones_like(X1)

    # True sigmoid parameters
    true_left_asymptote = 1.3
    true_right_asymptote = 9.5
    true_slope = np.array([-7, 1.6, 1.6])  # Slopes for both variables

    # Stack X1 and X2 to form a design matrix with two variables
    X_two_vars = np.column_stack([X0, X1, X2])

    # Generate Y data using the true sigmoid function
    Y_true = sigmoid(X_two_vars, true_left_asymptote, true_right_asymptote, true_slope)

    # Add some noise to the Y values
    noise = 0.75 * np.random.randn(len(X1))
    Y_noisy = Y_true.ravel() + noise

    return glm, X_two_vars, Y_noisy


def test_interactor_plot_input_validation():
    # Invalid DataFrame
    with pytest.raises(ValueError):
        _ = InteractorDiagnosticPlot(
            df=pd.DataFrame(np.random.rand(100, 2)),  # Only 2 columns
            quantile=0.1,
            B=np.array([1.0, 1.0, 1.0]),
        )

    # Invalid B (not a tuple of three floats)
    with pytest.raises(ValueError):
        _ = InteractorDiagnosticPlot(
            df=pd.DataFrame(np.random.rand(100, 3)),
            quantile=0.1,
            B=np.array([1.0, 1.0]),  # Only 2 elements
        )

    # Missing asymptotes for sigmoid model
    with pytest.raises(ValueError):
        _ = InteractorDiagnosticPlot(
            df=pd.DataFrame(np.random.rand(100, 3)),
            quantile=0.1,
            B=np.array([1.0, 1.0, 1.0]),
            model_type="sigmoid",
        )


def test_interactor_plot_linear_model(setup_glm_two_variable):
    glm, X_two_vars, Y_noisy = setup_glm_two_variable

    # Create a DataFrame
    df = pd.DataFrame(X_two_vars)

    # Test InteractorDiagnosticPlot for linear model
    plotter = InteractorDiagnosticPlot(
        df=df,
        quantile=0.1,
        B=np.array([1.0, 0.5, 0.2]),  # Example coefficients
        model_type="linear",
    )

    # Check that the plot method works
    plt_obj = plotter.plot()
    assert plt_obj is not None, "Plot for linear model should be generated"


def test_interactor_plot_sigmoid_model(setup_glm_two_variable):
    glm, X_two_vars, Y_noisy = setup_glm_two_variable

    # Create a DataFrame
    df = pd.DataFrame(X_two_vars)

    # Test InteractorDiagnosticPlot for sigmoid model
    plotter = InteractorDiagnosticPlot(
        df=df,
        quantile=0.1,
        B=np.array([1.0, 0.5, 0.2]),  # Example coefficients
        model_type="sigmoid",
        left_asymptote=0.5,
        right_asymptote=1.5,
    )

    # Check that the plot method works
    plt_obj = plotter.plot()
    assert plt_obj is not None, "Plot for sigmoid model should be generated"
