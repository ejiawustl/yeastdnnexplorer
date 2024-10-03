from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from yeastdnnexplorer.utils.sigmoid import sigmoid


class InteractorDiagnosticPlot:
    def __init__(
        self,
        df: pd.DataFrame,
        quantile: float,
        B: NDArray[np.float_],  # Accept a tuple of three coefficients
        model_type: Literal["linear", "sigmoid"] = "linear",
        font_size: int = 18,
        line_thickness: float = 3,
        legend_font_size: int = 12,
        axis_tick_font_size: int = 14,
        left_asymptote: float | None = None,
        right_asymptote: float | None = None,
    ):
        """Initialize the class with data and model parameters."""
        self.df = df
        self.quantile = quantile
        self.B = B  # Expect B as a tuple
        self.model_type = model_type
        self.font_size = font_size
        self.line_thickness = line_thickness
        self.legend_font_size = legend_font_size
        self.axis_tick_font_size = axis_tick_font_size
        self.left_asymptote = left_asymptote
        self.right_asymptote = right_asymptote

        # Validate inputs when initializing
        self.validate_inputs()

    def validate_inputs(self):
        """Validates the inputs for the class."""
        if not isinstance(self.df, pd.DataFrame) or self.df.shape[1] != 3:
            raise ValueError("`df` must be a DataFrame with exactly three columns.")
        if not (0 < self.quantile < 1):
            raise ValueError("Quantile must be between 0 and 1.")
        if len(self.B) != 3 or not all(isinstance(b, (float)) for b in self.B):
            raise ValueError("B must be a tuple with exactly three float values.")
        if self.model_type not in ["linear", "sigmoid"]:
            raise ValueError("model_type must be either 'linear' or 'sigmoid'.")
        if self.model_type == "sigmoid":
            if self.left_asymptote is None or self.right_asymptote is None:
                raise ValueError(
                    "You must provide left_asymptote and "
                    "right_asymptote for sigmoid model."
                )

    def compute_ecdf(self):
        """Computes the ECDF for the third column and splits the data into bottom and
        top quantiles."""
        self.df["ecdf"] = self.df.iloc[:, 2].rank(method="max") / len(self.df)
        self.df_bottom = self.df[self.df["ecdf"] <= self.quantile].copy()
        self.df_top = self.df[self.df["ecdf"] >= (1 - self.quantile)].copy()

        # Remove rows where X1 > 2.0 for cleaner plotting
        self.df_bottom = self.df_bottom[self.df_bottom.iloc[:, 1] <= 2.0]
        self.df_top = self.df_top[self.df_top.iloc[:, 1] <= 2.0]

    def calculate_means(self):
        """Calculates the mean of the interactor (X2) for bottom and top quantiles."""
        self.bottom_x2_mean = self.df_bottom.iloc[:, 2].mean()
        self.top_x2_mean = self.df_top.iloc[:, 2].mean()

    def add_jitter(self, data: np.ndarray, jitter_strength: float = 0.05) -> np.ndarray:
        """Adds jitter to a dataset to display point density."""
        return data + np.random.uniform(-jitter_strength, jitter_strength, len(data))

    def create_model_lines(self, **kwargs):
        """Creates model prediction lines for bottom and top quantiles."""
        self.x_vals = np.linspace(0, 2.0, 100)

        if self.model_type == "linear":
            # Linear model: B_0 + B_1 * x + B_2 * x * X2_mean
            self.bottom_line = (
                self.B[0]
                + self.B[1] * self.x_vals
                + self.B[2] * self.bottom_x2_mean * self.x_vals
            )
            self.top_line = (
                self.B[0]
                + self.B[1] * self.x_vals
                + self.B[2] * self.top_x2_mean * self.x_vals
            )

        elif self.model_type == "sigmoid":
            assert self.left_asymptote is not None and self.right_asymptote is not None

            x_lower = np.column_stack(
                (
                    np.ones_like(self.x_vals),
                    self.x_vals,
                    self.x_vals * self.bottom_x2_mean,
                )
            )
            x_upper = np.column_stack(
                (np.ones_like(self.x_vals), self.x_vals, self.x_vals * self.top_x2_mean)
            )

            # Compute the bottom and top model lines using sigmoid

            self.bottom_line = sigmoid(
                X=x_lower,
                left_asymptote=self.left_asymptote,
                right_asymptote=self.right_asymptote,
                B=self.B,
            )
            self.top_line = sigmoid(
                X=x_upper,
                left_asymptote=self.left_asymptote,
                right_asymptote=self.right_asymptote,
                B=self.B,
            )

        else:
            raise ValueError("Unsupported model_type. Use 'linear' or 'sigmoid'.")

    def create_plot(self, ax, x_data, y_data, line_data, color, label, mean_value):
        """Creates a scatter plot with a fitted line for a given quantile."""
        ax.scatter(x_data, y_data, color=color, alpha=0.5, label=label)
        ax.plot(
            self.x_vals,
            line_data,
            color=color,
            label=f"Model (Mean = {mean_value:.2f})",
            linewidth=self.line_thickness,
        )
        ax.axhline(0, color="black", linestyle="--", linewidth=self.line_thickness)
        ax.tick_params(axis="both", which="major", labelsize=self.axis_tick_font_size)
        ax.legend(fontsize=self.legend_font_size, loc="upper left", frameon=False)
        ax.set_xlabel("CBF1 binding strength", fontsize=self.font_size)
        ax.set_ylabel("CBF1 perturbation response", fontsize=self.font_size)

    def plot(self, **kwargs):
        """Main plotting method that returns the `plt` object for further
        customization."""
        if self.model_type == "sigmoid":
            for k in ["left_asymptote", "right_asymptote", "B"]:
                if k in kwargs:
                    raise ValueError(
                        f"You must provide {k} as a keyword argument for sigmoid model."
                    )
        # Compute ECDF and split data
        self.compute_ecdf()

        # Calculate means of interactor (X2) for both quantiles
        self.calculate_means()

        # Create model lines for the plots
        self.create_model_lines(**kwargs)

        # Add jitter to Y values
        self.df_bottom.iloc[:, 0] = self.add_jitter(self.df_bottom.iloc[:, 0])
        self.df_top.iloc[:, 0] = self.add_jitter(self.df_top.iloc[:, 0])

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot bottom quantile
        self.create_plot(
            ax1,
            self.df_bottom.iloc[:, 1],
            self.df_bottom.iloc[:, 0],
            self.bottom_line,
            "blue",
            "Bottom 10% by Met31 binding",
            self.bottom_x2_mean,
        )

        # Plot top quantile
        self.create_plot(
            ax2,
            self.df_top.iloc[:, 1],
            self.df_top.iloc[:, 0],
            self.top_line,
            "red",
            "Top 10% by Met31 binding",
            self.top_x2_mean,
        )

        # Return plt object for further customization
        return plt

    def __call__(self, **kwargs):
        """Allows the object to be called like a function, invoking the `plot`
        method."""
        return self.plot(**kwargs)
