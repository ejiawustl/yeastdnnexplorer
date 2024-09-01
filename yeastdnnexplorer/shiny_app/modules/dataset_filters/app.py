"""
The dataset_filters/expression module adds a multi-select input for selecting expression
assays (eg tfko, overexpression).

Based on the selection, it adds/removes additional
filtering options. app.py provides an example of how to use
the dataset_filters/expression module.
Launch with
`scre`

"""

import logging

from shiny import App, run_app, ui

from yeastdnnexplorer.shiny_app.modules.dataset_filters.module import (
    dataset_filters_server,
    dataset_filters_ui,
)
from yeastdnnexplorer.utils import configure_logger

logger = logging.getLogger("shiny")

# Call the logger configuration function
configure_logger("shiny")

app_ui = ui.page_fluid(ui.layout_sidebar(dataset_filters_ui("data_filters")))


def app_server(input, output, session):
    dataset_filters_server("data_filters")


# Create an app instance
app = App(ui=app_ui, server=app_server)


if __name__ == "__main__":
    run_app(
        "yeastdnnexplorer.shiny_app.modules.dataset_filters.app:app",
        reload=True,
        reload_dirs=["."],
    )
