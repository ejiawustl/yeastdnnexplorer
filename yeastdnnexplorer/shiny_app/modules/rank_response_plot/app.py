import logging

from shiny import App, run_app, ui

from yeastdnnexplorer.shiny_app.modules.rank_response_plot.module import (
    rank_response_plot_server,
    rank_response_plot_ui,
)
from yeastdnnexplorer.utils import configure_logger

logger = logging.getLogger("shiny")

# Call the logger configuration function
configure_logger("shiny")

app_ui = ui.page_fluid(rank_response_plot_ui("rank_response_plot"))


def app_server(input, output, session):
    # No need to make this async; `rank_response_plot_server`
    # handles async tasks internally
    rank_response_plot_server("rank_response_plot")


# Create an app instance
app = App(ui=app_ui, server=app_server)

if __name__ == "__main__":
    run_app(
        "yeastdnnexplorer.shiny_app.modules.rank_response_plot.app:app",
        reload=True,
        reload_dirs=["."],
    )
