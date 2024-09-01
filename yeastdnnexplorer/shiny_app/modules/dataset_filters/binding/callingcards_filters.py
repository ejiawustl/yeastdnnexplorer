import logging
from typing import Any

from shiny import module, ui

logger = logging.getLogger("shiny")


@module.ui
def callingcards_filter_ui(row_id="callingcards_filters_rows"):
    return ui.row(
        ui.card(
            ui.card_header("Calling Cards Filters", class_="filter_card_header"),
            ui.card(
                ui.card_header("Lab Selection", class_="filter_card_header"),
                ui.input_checkbox_group(
                    "lab",
                    "",
                    choices=["brent", "mitra"],
                    selected=["brent", "mitra"],
                    inline=True,
                ),
                style="border: 1px solid #ccc; margin-bottom: 15px;",
            ),
            ui.card(
                ui.card_header("Replicate Selection", class_="filter_card_header"),
                ui.p(
                    "The callingcards data has replicates. "
                    + "Choose 'single' or 'combined' "
                    + "to return only the replicates, "
                    + "or only the combined replicate data. "
                    + "Select both to return all."
                ),
                ui.input_checkbox_group(
                    "combined_replicates",
                    "",
                    choices=["single", "combined"],
                    selected=["combined"],
                    inline=True,
                ),
                style="border: 1px solid #ccc; margin-bottom: 15px;",
            ),
            ui.card(
                ui.card_header("Data Usability", class_="filter_card_header"),
                ui.input_switch("data_usable", "Data Usable", True),
                style="border: 1px solid #ccc; margin-bottom: 15px;",
            ),
        ),
        id=row_id,
    )


@module.server
def callingcards_filter_server(input: Any, output: Any, session: Any):
    logger.debug("callingcards_filter server")
    return input.lab, input.combined_replicates, input.data_usable
