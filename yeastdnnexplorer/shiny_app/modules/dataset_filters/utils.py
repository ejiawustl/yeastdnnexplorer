import logging
from collections.abc import Callable
from typing import Any

from shiny import reactive, req, ui

logger = logging.getLogger("shiny")


def create_dynamic_filter_ui(
    assay_selector_reactive: reactive.Value,
    assay_name: str,
    insert_selector: str,
    filter_ui_module: Callable,
    filter_ui_id: str,
    row_id: str,
    displayed_assay_current_state: reactive.Value,
    **kwargs: dict[str, Any],
) -> None:
    """
    This is intended to be used to add/remove filtering options based on the selected
    assays for the binding and expression filter modules. If the filter, eg the
    'harbison_filter' takes a set of items that are extracted from a reactive value,
    that reactive value can be passed in the kwargs. The key should be the argument name
    in the ui module and the value should be the reactive. The reactive will be .get()
    before passing to the ui module.

    :param assay_name: The name of the assay to add/remove the filter for.
    :type assay_name: str
    :param insert_selector: The selector to insert the UI before
    :type insert_selector: str
    :param filter_ui_module: The UI module to insert/remove
    :type filter_ui_module: Callable
    :param filter_ui_id: The ID of the filter UI
    :type filter_ui_id: str
    :param row_id: The ID of the row to insert the UI into
    :type row_id: str
    :param displayed_assay_current_state: The reactive value to track the displayed
        assays
    :type displayed_assay_current_state: reactive.Value
    :param kwargs: Additional arguments to pass to the UI module
    :type kwargs: Dict[str, Any]

    """

    @reactive.effect()
    @reactive.event(assay_selector_reactive)
    async def _():
        """Add/remove filtering options based on the selected assays."""
        if (
            assay_name in assay_selector_reactive.get()
            and assay_name in displayed_assay_current_state.get()
        ):
            logger.debug(f"{assay_name} already in displayed_assay_current_state")
            pass
        elif (
            assay_name in assay_selector_reactive.get()
            and assay_name not in displayed_assay_current_state.get()
        ):
            if kwargs:
                d = kwargs.copy()
                for k, v in d.items():
                    if isinstance(v, reactive.Value):
                        kwargs[k] = v.get()

            # Insert the UI for the filter
            ui.insert_ui(
                filter_ui_module(
                    filter_ui_id,
                    row_id=row_id,
                    **kwargs,  # Pass additional arguments to the UI module
                ),
                selector="#" + insert_selector,
                where="beforeEnd",
            )

            # Update the state tracker
            s = displayed_assay_current_state.get()
            s.add(assay_name)
            displayed_assay_current_state.set(s)

        elif assay_name in displayed_assay_current_state.get():
            logger.debug(f"Removing {assay_name} filter")
            ui.remove_ui(selector=f"#{row_id}")

            # Update the state tracker
            s = displayed_assay_current_state.get()
            s.remove(assay_name)
            displayed_assay_current_state.set(s)


# Loop through the mapping and create reactive effects to update the dictionary
def update_outer_scope_reactive_dict(
    outer_scope_reactive_dict, inner_scope_reactive_dict
) -> None:
    """
    The data_filters module is designed to take dictionaries of reactive values for the
    binding and expression filters. The binding/expression module servers output the
    result of the users selections as tuples. In the data_filterset server, those
    reactives are collected into a dictionary, the `inner_scope_reactive_dict`, of the
    same structure as the `outer_scope_reactive_dict`, which is passed as input to the
    module server. The inner scope reactive dict is used to modify the outer scope
    reactive dict. The structure of both dictionaries is expected to be either a
    key/value pair, or a key/dictionary pair. there should be only 1 level of nesting.
    The dictionaries are required to have the same keys.

    :param outer_scope_reactive_dict: The dictionary of reactive values to update
    :type outer_scope_reactive_dict: dict
    :param inner_scope_reactive_dict: The mapping of reactive values to update the
        dictionary
    :type inner_scope_reactive_dict: dict
    :raises KeyError: If the keys of the input dictionaries are not the same
    :raises ValueError: If the inner scope reactive dict does not have reactive values
        as values

    """
    # verify that the keys of the input dictionaries are the same
    if outer_scope_reactive_dict.keys() != inner_scope_reactive_dict.keys():
        raise KeyError(
            "The outer and inner scope reactive dictionaries must have the same keys"
        )
    for outer_key, outer_value in inner_scope_reactive_dict.items():
        if isinstance(outer_value, reactive.Value):

            @reactive.effect()
            def _update_reactive(
                outer_scope_reactive_dict=outer_scope_reactive_dict,
                outer_key=outer_key,
                outer_value=outer_value,
            ):
                req(outer_value)
                outer_scope_reactive_dict[outer_key].set(outer_value.get())

        elif isinstance(outer_value, dict):
            if outer_value.keys() != outer_scope_reactive_dict[outer_key].keys():
                raise KeyError(
                    "The inner scope reactive dict must have the same keys as the "
                    + "outer scope reactive dict"
                )
            for inner_key, inner_value in outer_value.items():
                if isinstance(inner_value, reactive.Value):

                    @reactive.effect()
                    def _update_reactive(
                        outer_scope_reactive_dict=outer_scope_reactive_dict,
                        outer_key=outer_key,
                        inner_key=inner_key,
                        inner_value=inner_value,
                    ):
                        req(inner_value)
                        outer_scope_reactive_dict[outer_key][inner_key].set(
                            inner_value.get()
                        )

                else:
                    raise ValueError(
                        "The inner scope reactive dict must have "
                        + "reactive values as values"
                    )
