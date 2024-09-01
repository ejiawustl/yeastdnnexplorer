import logging

import pandas as pd
from shiny import Inputs, Outputs, Session, module, reactive, req, ui

from yeastdnnexplorer.interface import BindingAPI, ExpressionAPI
from yeastdnnexplorer.shiny_app.modules.dataset_filters.binding import (
    module as binding_module,
)
from yeastdnnexplorer.shiny_app.modules.dataset_filters.expression import (
    module as expression_module,
)
from yeastdnnexplorer.shiny_app.modules.dataset_filters.utils import (
    update_outer_scope_reactive_dict,
)

logger = logging.getLogger("shiny")


@module.ui
def dataset_filters_ui():
    return ui.sidebar(
        binding_module.dataset_selector_ui("binding_data_filters"),
        expression_module.dataset_selector_ui("expression_data_filters"),
        width=500,
    )


@module.server
def dataset_filters_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    binding_reactives: dict[str, reactive.Value | dict[str, reactive.Value]],
    expression_reactives: dict[str, reactive.Value | dict[str, reactive.Value]],
    upset_reactives: dict[str, reactive.Value],
) -> None:
    # Binding Setup
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

    binding_server_reactive_dict = binding_module.dataset_selector_server(
        "binding_data_filters",
        binding_assay_options=_binding_assay_options,
        harbison_conditions_options=_harbison_conditions_options,
    )

    update_outer_scope_reactive_dict(binding_reactives, binding_server_reactive_dict)

    # Expression Setup
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

    expression_server_reactive_mapping = expression_module.dataset_selector_server(
        "expression_data_filters",
        expression_assay_options=_expression_assay_options,
        mcisaac_mechanism_options=_mcisaac_mechanism_options,
        mcisaac_restriction_options=_mcisaac_restriction_options,
        mcisaac_time_options=_mcisaac_time_options,
        mcisaac_replicate_options=_mcisaac_replicate_options,
    )

    update_outer_scope_reactive_dict(
        expression_reactives, expression_server_reactive_mapping
    )
