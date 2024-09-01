"""
The dataset_filters/binding module adds a multi-select input for selecting binding
assays (eg callingcards, chipexo, chip).

Based on the selection, it adds/removes additional
filtering options. app.py provides an example of how to use the
dataset_filters/binding module.
Launch with
`poetry run python yeastdnnexplorer/shiny_app/modules/dataset_filters/binding/app.py`

"""

import logging

import pandas as pd
from shiny import App, reactive, req, run_app, ui

from yeastdnnexplorer.interface import BindingAPI
from yeastdnnexplorer.shiny_app.modules.dataset_filters.binding import (
    dataset_selector_server,
    dataset_selector_ui,
)
from yeastdnnexplorer.utils import configure_logger

logger = logging.getLogger("shiny")

# Call the logger configuration function
configure_logger("shiny")

app_ui = ui.page_sidebar(ui.sidebar(dataset_selector_ui("binding_data_filters")))


def app_server(input, output, session):
    # Initialize the API (assuming some default configuration)
    _bindingAPI = reactive.Value(BindingAPI())
    _binding_metadata_df = reactive.Value()
    _binding_assay_options = reactive.Value()
    _harbison_conditions_options = reactive.Value()

    @reactive.effect()
    async def _():
        """Set/update the _binding_metadata_df reactive value with the current
        metadata."""
        req(_bindingAPI)
        logger.info("Fetching BindingAPI metadata")
        binding_api = _bindingAPI.get()
        res = await binding_api.read()
        try:
            df = res.get("metadata", pd.DataFrame())
            _binding_metadata_df.set(df)
            logger.debug(f"_binding_metadata_df: {df.head()}")
        except KeyError:
            logger.error(
                "Could not retrieve BindingAPI metadata. "
                + "The 'metadata' key was not found in the binding API response."
            )
        except (ValueError, TimeoutError, pd.errors.EmptyDataError) as exc:
            logger.error(
                f"An error occurred while fetching the "
                f"BindingAPI metadata. Error: {exc}"
            )

    @reactive.effect()
    async def _():
        """Set/update the _binding_assay_options reactive value with the available
        binding assays."""
        req(_binding_metadata_df)
        logger.info("Fetching binding assay list")
        df = _binding_metadata_df.get()
        try:
            binding_assays = df.assay.unique().tolist()
            _binding_assay_options.set(binding_assays)
            logger.debug(f"Binding assays: {binding_assays}")
        except AttributeError as exc:
            logger.error(
                f"The 'assay' column was not found in the binding metadata. "
                f"Error: {exc}"
            )

    @reactive.effect()
    async def _():
        """Set/update the _harbison_conditions_options reactive value with the current
        Harbison conditions."""
        req(_binding_metadata_df)
        logger.info("Fetching Harbison conditions")
        df = _binding_metadata_df.get()
        try:
            harbison_choices = (
                df[df.source_name == "harbison_chip"].condition.unique().tolist()
            )
            _harbison_conditions_options.set(harbison_choices)
            logger.debug(f"Harbison conditions: {harbison_choices}")
        except AttributeError as exc:
            logger.error(
                f"The 'condition' column was not found in the binding "
                f"metadata DataFrame. Error: {exc}"
            )

    dataset_selector_server(
        "binding_data_filters",
        binding_assay_options=_binding_assay_options,
        harbison_conditions_options=_harbison_conditions_options,
    )


# Create an app instance
app = App(ui=app_ui, server=app_server)


if __name__ == "__main__":
    run_app(
        "yeastdnnexplorer.shiny_app.modules.dataset_filters.binding.app:app",
        reload=True,
        reload_dirs=["."],
    )
