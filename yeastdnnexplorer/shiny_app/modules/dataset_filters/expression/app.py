"""
The dataset_filters/expression module adds a multi-select input for selecting expression
assays (eg tfko, overexpression).

Based on the selection, it adds/removes additional
filtering options. app.py provides an example of how to use
the dataset_filters/expression module.
Launch with
`poetry run python yeastdnnexplorer/shiny_app/modules/dataset_filters/expression/app.py`

"""

import logging

import pandas as pd
from shiny import App, reactive, req, run_app, ui

from yeastdnnexplorer.interface import ExpressionAPI
from yeastdnnexplorer.shiny_app.modules.dataset_filters.expression import (
    dataset_selector_server,
    dataset_selector_ui,
)
from yeastdnnexplorer.utils import configure_logger

logger = logging.getLogger("shiny")

# Call the logger configuration function
configure_logger("shiny")

app_ui = ui.page_fluid(
    dataset_selector_ui("expression_data_filters"),
)


def app_server(input, output, session):
    # Initialize the API (assuming some default configuration)
    _expressionAPI = reactive.Value(ExpressionAPI())
    _expression_metadata_df = reactive.Value()
    _expression_assay_options = reactive.Value()
    _mcisaac_mechanism_options = reactive.Value()
    _mcisaac_restriction_options = reactive.Value()
    _mcisaac_time_options = reactive.Value()
    _mcisaac_replicate_options = reactive.Value()

    @reactive.effect()
    async def _():
        """Set/update the _expression_metadata_df reactive value with the current
        metadata."""
        req(_expressionAPI)
        logger.info("Fetching ExpressionAPI metadata")
        expression_api = _expressionAPI.get()
        res = await expression_api.read()
        try:
            df = res.get("metadata", pd.DataFrame())
            _expression_metadata_df.set(df)
            logger.debug(f"_expression_metadata_df: {df.head()}")
        except KeyError:
            logger.error(
                "Could not retrieve ExpressionAPI metadata. "
                + "The 'metadata' key was not found in the expression API response."
            )
        except (ValueError, TimeoutError, pd.errors.EmptyDataError) as exc:
            logger.error(
                f"An error occurred while fetching the "
                f"ExpressionAPI metadata. Error: {exc}"
            )

    @reactive.effect()
    async def _():
        """Set/update the _expression_assay_options reactive value with the available
        expression assays."""
        req(_expression_metadata_df)
        logger.info("Fetching expression assay list")
        df = _expression_metadata_df.get()
        try:
            expression_assays = df.assay.unique().tolist()
            _expression_assay_options.set(expression_assays)
            logger.debug(f"Binding assays: {expression_assays}")
        except AttributeError as exc:
            logger.error(
                f"The 'assay' column was not found in the expression metadata. "
                f"Error: {exc}"
            )

    @reactive.effect()
    async def _():
        """Set/update the mcisaac option reactives: mechanism, restriction, time,
        replicate."""
        req(_expression_metadata_df)
        logger.info("Fetching Mcisaac OE conditions")
        df = _expression_metadata_df.get()
        try:
            mcisaac_df = df[df.source_name == "mcisaac_oe"]
        except AttributeError as exc:
            logger.error(
                f"The 'source_name' column was not found in the expression "
                f"metadata DataFrame. Error: {exc}"
            )
        try:
            mechanism_options = mcisaac_df.mechanism.unique().tolist()
            _mcisaac_mechanism_options.set(mechanism_options)
            logger.debug(f"McIsaac mechanism options: {mechanism_options}")
        except AttributeError as exc:
            logger.error(
                f"The 'mechanism' column was not found in the expression "
                f"metadata DataFrame. Error: {exc}"
            )
        try:
            restriction_options = mcisaac_df.restriction.unique().tolist()
            _mcisaac_restriction_options.set(restriction_options)
            logger.debug(f"McIsaac restriction options: {restriction_options}")
        except AttributeError as exc:
            logger.error(
                f"The 'restriction' column was not found in the expression "
                f"metadata DataFrame. Error: {exc}"
            )
        try:
            time_options = mcisaac_df.time.unique().tolist()
            time_options.sort()
            time_options = [str(x) for x in time_options]
            _mcisaac_time_options.set(time_options)
            logger.debug(f"McIsaac time options: {time_options}")
        except AttributeError as exc:
            logger.error(
                f"The 'time' column was not found in the expression "
                f"metadata DataFrame. Error: {exc}"
            )
        try:
            replicate_options = mcisaac_df.replicate.unique().tolist()
            _mcisaac_replicate_options.set(replicate_options)
            logger.debug(f"McIsaac replicate options: {replicate_options}")
        except AttributeError as exc:
            logger.error(
                f"The 'replicate' column was not found in the expression "
                f"metadata DataFrame. Error: {exc}"
            )

    dataset_selector_server(
        "expression_data_filters",
        expression_assay_options=_expression_assay_options,
        mcisaac_mechanism_options=_mcisaac_mechanism_options,
        mcisaac_restriction_options=_mcisaac_restriction_options,
        mcisaac_time_options=_mcisaac_time_options,
        mcisaac_replicate_options=_mcisaac_replicate_options,
    )


# Create an app instance
app = App(ui=app_ui, server=app_server)


if __name__ == "__main__":
    run_app(
        "yeastdnnexplorer.shiny_app.modules.dataset_filters.expression.app:app",
        reload=True,
        reload_dirs=["."],
    )
