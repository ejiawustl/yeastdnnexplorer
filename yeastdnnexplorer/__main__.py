import argparse
import json
import logging
import os
import time
from typing import Literal

import joblib
import pandas as pd
from shiny import run_app
from sklearn.linear_model import LassoCV

from yeastdnnexplorer.ml_models.lasso_modeling import (
    bootstrap_stratified_cv_modeling,
    generate_modeling_data,
    stratified_cv_modeling,
)
from yeastdnnexplorer.utils import LogLevel, configure_logger

logger = logging.getLogger("main")


def configure_logging(
    log_level: int, handler_type: Literal["console", "file"] = "console"
) -> tuple[logging.Logger, logging.Logger]:
    """
    Configure the logging for the application.

    :param log_level: The logging level to set.
    :return: A tuple of the main and shiny loggers.

    """
    # add a timestamp to the log file name
    log_file = f"yeastdnnexplorer_{time.strftime('%Y%m%d-%H%M%S')}.log"
    main_logger = configure_logger(
        "main", level=log_level, handler_type=handler_type, log_file=log_file
    )
    shiny_logger = configure_logger(
        "shiny", level=log_level, handler_type=handler_type, log_file=log_file
    )
    return main_logger, shiny_logger


def run_shiny(args: argparse.Namespace) -> None:
    """
    Run the shiny app with the specified arguments.

    :param args: The parsed command-line arguments.

    """
    kwargs = {}
    if args.debug:
        kwargs["reload"] = True
        kwargs["reload_dirs"] = ["yeastdnnexplorer/shiny_app"]  # type: ignore
    app_import_string = "yeastdnnexplorer.shiny_app.app:app"
    run_app(app_import_string, **kwargs)


# this goes along with an example in the arg parser below, showing how to
# add cmd line utilies
# def run_another_command(args: argparse.Namespace) -> None:
#     """
#     Run another command with the specified arguments.

#     :param args: The parsed command-line arguments.
#     """
#     print(f"Running another command with parameter: {args.param}")


def run_lasso_bootstrap(args: argparse.Namespace) -> None:
    """
    Run LassoCV with bootstrap resampling on a specified transcription factor.

    :param args: The parsed command-line arguments.

    """
    output_dirpath = os.path.join(args.output_dir, args.perturbed_tf)
    if os.path.exists(output_dirpath):
        raise FileExistsError(
            f"File {args.output_dir} already exists. "
            "Please specify a different `output_dir`."
        )
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        # Ensure the entire output directory path exists
        os.makedirs(output_dirpath, exist_ok=True)
    if not os.path.exists(args.response_file):
        raise FileNotFoundError(f"File {args.response_file} does not exist.")
    if not os.path.exists(args.predictors_file):
        raise FileNotFoundError(f"File {args.predictors_file} does not exist.")

    # Load data
    Y_filtered_transformed = pd.read_csv(args.response_file, index_col=0)
    predictors_df = pd.read_csv(args.predictors_file, index_col=0)

    # Generate modeling data
    y, X = generate_modeling_data(
        colname=args.perturbed_tf,
        response_df=Y_filtered_transformed,
        predictors_df=predictors_df,
        drop_intercept=True,
        quantile_threshold=args.data_quantile,
        formula=args.formula,
    )

    # Configure and fit LassoCV estimator
    lassoCV_estimator = LassoCV(
        fit_intercept=True,
        max_iter=10000,
        selection="random",
        random_state=42,
        n_jobs=4,
    )

    # Fit the model to extract alphas
    try:
        lasso_model = stratified_cv_modeling(y, X, lassoCV_estimator)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to fit the LassoCV model on {args.perturbed_tf}."
        ) from exc

    # Set the alphas of the main estimator for bootstrap consistency
    lassoCV_estimator.alphas_ = lasso_model.alphas_

    # Run bootstrap modeling
    try:
        bootstrap_output = bootstrap_stratified_cv_modeling(
            y=y,
            X=X,
            estimator=lassoCV_estimator,
            bootstrap_cv=True,
            n_bootstraps=args.n_bootstraps,
            ci_percentiles=[90.0, 95.0, 99.0, 100.0],
            max_iter=10000,
            fit_intercept=True,
            selection="random",
            random_state=42,
            drop_columns_before_modeling=args.drop_columns_before_modeling,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to run bootstrap modeling on {args.perturbed_tf}."
        ) from exc

    # the lasso_model and bootstrap_output in output_dirpath. boostrap_output is a
    # is a tuple where the first value is a dictionary called "ci_dict",
    # the second is a data frame called "bootstrap_coef_df",
    # and the third is a list called "bootstrap_alphas".

    # Save the fitted Lasso model
    lasso_model_path = os.path.join(output_dirpath, "lasso_model.joblib")
    joblib.dump(lasso_model, lasso_model_path)

    # Save ci_dict as JSON
    ci_dict = bootstrap_output[0]
    ci_dict_path = os.path.join(output_dirpath, "ci_dict.json")
    with open(ci_dict_path, "w") as f:
        json.dump(ci_dict, f, indent=4)

    # Save bootstrap_coef_df as CSV
    bootstrap_coef_df = bootstrap_output[1]
    bootstrap_coef_df_path = os.path.join(output_dirpath, "bootstrap_coef_df.csv")
    bootstrap_coef_df.to_csv(bootstrap_coef_df_path, index=False)

    # Save bootstrap_alphas as CSV
    bootstrap_alphas = pd.DataFrame(bootstrap_output[2], columns=["alpha"])
    bootstrap_alphas_path = os.path.join(output_dirpath, "bootstrap_alphas.csv")
    bootstrap_alphas.to_csv(bootstrap_alphas_path, index=False)


class CustomHelpFormatter(argparse.HelpFormatter):
    """Custom help formatter to format the subcommands and general options sections."""

    # CustomHelpFormatter code remains the same as you provided


def add_general_arguments_to_subparsers(subparsers, general_arguments):
    for subparser in subparsers.choices.values():
        for arg in general_arguments:
            subparser._add_action(arg)


def main() -> None:
    """Main entry point for the YeastDNNExplorer application."""
    parser = argparse.ArgumentParser(
        prog="yeastdnnexplorer",
        description="YeastDNNExplorer Main Entry Point",
        usage="yeastdnnexplorer --help",
        formatter_class=CustomHelpFormatter,
    )

    formatter = parser._get_formatter()

    # Shared parameter for logging level
    log_level_argument = parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    log_handler_argument = parser.add_argument(
        "--log-handler",
        type=str,
        default="console",
        choices=["console", "file"],
        help="Set the logging handler",
    )
    formatter.add_arguments([log_level_argument, log_handler_argument])

    # Define subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Shiny command
    shiny_parser = subparsers.add_parser(
        "shiny",
        help="Run the shiny app",
        description="Run the shiny app",
        formatter_class=CustomHelpFormatter,
    )
    shiny_parser.add_argument(
        "--debug", action="store_true", help="Run the app with reloading enabled"
    )
    shiny_parser.set_defaults(func=run_shiny)

    # An example of adding another command
    # another_parser = subparsers.add_parser(
    #     "another_command",
    #     help="Run another command",
    #     description="Run another command",
    #     formatter_class=CustomHelpFormatter,
    # )
    # another_parser.add_argument(
    #     "--param", type=str, required=True, help="A parameter for another command"
    # )
    # another_parser.set_defaults(func=run_another_command)

    # Lasso Bootstrap command
    lasso_parser = subparsers.add_parser(
        "lasso_bootstrap",
        help="Run LassoCV with bootstrap resampling",
        description="Run LassoCV with bootstrap resampling",
        formatter_class=CustomHelpFormatter,
    )

    # Input arguments
    input_group = lasso_parser.add_argument_group("Input")
    input_group.add_argument(
        "--response_file",
        type=str,
        required=True,
        help="Path to the response CSV file. NOTE: the index column must be "
        "present and the first column in the CSV. Additionally, the index values "
        "should be either symbols or locus tags, matching the index values in both "
        "response and predictors files. The perturbed gene will be removed from "
        "the model data only if column names in response data match the index "
        "format (e.g., symbol or locus tag).",
    )
    input_group.add_argument(
        "--predictors_file",
        type=str,
        required=True,
        help="Path to the predictors CSV file. NOTE: the index column must be "
        "present and the first column in the CSV. Additionally, the index values "
        "should be either symbols or locus tags, matching the index values in both "
        "response and predictors files. The perturbed gene will be removed from the "
        "model data only if column names in predictors data match the index "
        "format (e.g., symbol or locus tag).",
    )
    input_group.add_argument(
        "--perturbed_tf",
        type=str,
        help="A response variable column to use. The data indices should be in "
        "the same format (e.g., symbol or locus tag) so that the perturbed gene can "
        "be removed from the data prior to modeling.",
    )
    input_group.add_argument(
        "--formula",
        type=str,
        default=None,
        help="The formula to use for modeling. If omitted, a formula with all of "
        "the interactors will be used, eg "
        "perturbed_tf_lrr ~ perturbed_tf + perturbed_tf:other_tf1 + ...",
    )
    input_group.add_argument(
        "--drop_columns_before_modeling",
        type=str,
        nargs="+",
        default=[],
        help="List of columns to drop from the predictors data before modeling. "
        "This is useful if you need to include a column for the stratification "
        "but do not want to use it as a predictor",
    )
    input_group.add_argument(
        "--data_quantile",
        type=float,
        default=None,
        help="The quantile threshold for filtering the data based on the "
        "perturbed binding data. For example, 0.1 would select the top 10 percent. "
        "If omitted, all data will be used.",
    )
    input_group.add_argument(
        "--n_bootstraps",
        type=int,
        default=1000,
        help="Number of bootstrap samples to generate.",
    )

    # Output arguments
    output_group = lasso_parser.add_argument_group("Output")
    output_group.add_argument(
        "--output_dir",
        type=str,
        default="./lasso_bootstrap_output",
        help="Path to the output directory where results will be saved.",
    )

    lasso_parser.set_defaults(func=run_lasso_bootstrap)

    # Add the general arguments to the subcommand parsers
    add_general_arguments_to_subparsers(subparsers, [log_level_argument])

    # Parse arguments
    args = parser.parse_args()

    # Configure logging
    try:
        log_level = LogLevel.from_string(args.log_level)
    except ValueError as e:
        print(e)
        parser.print_help()
        return

    main_logger, shiny_logger = configure_logging(log_level)

    # Run the appropriate command
    if args.command is None:
        parser.print_help()
    else:
        args.func(args)


if __name__ == "__main__":
    main()
