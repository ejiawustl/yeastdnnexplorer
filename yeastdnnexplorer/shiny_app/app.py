import logging

from shiny import App, reactive, run_app, ui

from yeastdnnexplorer.shiny_app.modules.dataset_filters.module import (
    dataset_filters_server,
    dataset_filters_ui,
)
from yeastdnnexplorer.shiny_app.modules.rank_response_plot.module import (
    rank_response_plot_server,
    rank_response_plot_ui,
)
from yeastdnnexplorer.shiny_app.modules.upset_plot.module import (
    upset_plot_server,
    upset_plot_ui,
)
from yeastdnnexplorer.utils import configure_logger

logger = logging.getLogger("shiny")

# Call the logger configuration function
configure_logger("shiny", level=logging.INFO)

app_ui = ui.page_sidebar(
    dataset_filters_ui("data_filters"),
    upset_plot_ui("upset_plot"),
    rank_response_plot_ui("rank_response_plot"),
)


def app_server(input, output, session):
    _upset_reactives = {
        "sets": reactive.Value(),
        "regulators": reactive.Value(),
    }
    _binding_reactives = {
        "assay": reactive.Value(),
        "callingcards": {
            "lab": reactive.Value(),
            "combined_replicates": reactive.Value(),
            "data_usable": reactive.Value(),
        },
        "harbison": {
            "conditions": reactive.Value(),
        },
    }
    _expression_reactives = {
        "assay": reactive.Value(),
        "mcisaac": {
            "mechanism": reactive.Value(),
            "effect_colname": reactive.Value(),
            "restriction": reactive.Value(),
            "time": reactive.Value(),
            "replicate": reactive.Value(),
            "data_usable": reactive.Value(),
        },
        "tfko": {
            "source": reactive.Value(),
            "replicate": reactive.Value(),
            "data_usable": reactive.Value(),
        },
    }

    dataset_filters_server(
        "data_filters", _binding_reactives, _expression_reactives, _upset_reactives
    )
    upset_plot_server(
        "upset_plot", _binding_reactives, _expression_reactives, _upset_reactives
    )
    rank_response_plot_server("rank_response_plot")


# Create an app instance
app = App(ui=app_ui, server=app_server)

if __name__ == "__main__":
    run_app(
        "yeastdnnexplorer.shiny_app.app:app",
        reload=True,
        reload_dirs=["."],
    )

# deprecated snippet showing how to use the output to hide items in the UI
# import logging

# # from plotly import express as px
# from shiny import Inputs, Outputs, Session, reactive, render
# from shinywidgets import render_widget
# from upsetjs_jupyter_widget import UpSetJSWidget

# from ..modules import login_interface_server

# logger = logging.getLogger("shiny")

# # Sample penguin data
# penguins = {
#     "body_mass_g": [
#         3750,
#         3800,
#         3250,
#         3700,
#         3450,
#         3650,
#         3625,
#         3575,
#         3675,
#         3450,
#         3775,
#         3700,
#         3775,
#         4100,
#         3950,
#         3650,
#         3900,
#         4000,
#         4550,
#         4250,
#     ]
# }


# def server(input: Inputs, output: Outputs, session: Session):
#     _authenticated, _token = login_interface_server("authenticate")
#     # Define a reactive value to store the selected items
#     _selected_items = reactive.Value(None)

#     @output
#     @render_widget()
#     def rank_response_overview_plot():
#         # generate_rank_response_overview_plot()

#         scatterplot = px.histogram(
#             data_frame=penguins,
#             x="body_mass_g",
#             nbins=input.n(),
#         ).update_layout(
#             title={"text": "Penguin Mass", "x": 0.5},
#             yaxis_title="Count",
#             xaxis_title="Body Mass (g)",
#         )
#         return scatterplot

#     @output
#     @render.text
#     def authenticated():
#         logger.debug(f"authenticated: {_authenticated.get()}")
#         return "true" if _authenticated.get() else "false"

#     @output
#     @render_widget
#     def upsetjs_plot():
#         # Create the UpSetJSWidget instance
#         w = UpSetJSWidget[str]()

#         # Populate the widget with data
#         w.from_dict(
#             dict(
#                 b_callingcards=["a", "b", "d", "e", "j"],
#                 b_harbison=["a", "b", "c", "e", "g", "h", "k", "l", "m"],
#                 b_chipexo=["a", "e", "f", "g", "h", "i", "j", "l", "m"],
#                 p_mcisaac=["a", "b", "c", "g", "h", "i", "j", "k", "l", "m"],
#                 p_kemmeren=["a", "e", "f", "g", "h", "i", "j", "k", "l", "m"],
#                 p_hu_reiman=["a", "b", "c", "d", "e", "f", "k", "l", "m"],
#             ),
#             order_by="name",
#         )

#         # Define a function to capture selection changes
#         def selection_changed(s):
#             _selected_items.set(s if s else None)

#         # Attach the callback to the widget's selection change event
#         w.on_selection_changed(selection_changed)

#         return w

#     # Example of how to use the _selected_items reactive variable
#     # elsewhere in your app
#     @output
#     @render.text
#     def selected_items():
#         return f"Selected Items: {_selected_items.get()}"


# from shiny import ui
# from shinywidgets import output_widget

# from ..modules import login_interface_ui

# app_ui = ui.page_fluid(
#     # this div is used to populate the output.authenticated in the UI to allow
#     # control over visibility of the login form and the main content
#     ui.div(
#         {
#             "style": "position: absolute; clip: rect(0,0,0,0); "
#             + "width: 0; height: 0; margin: 0; padding: 0; "
#             + "border: 0; overflow: hidden;"
#         },
#         ui.output_text_verbatim("authenticated"),
#     ),
#     ui.panel_conditional(
#         "output.authenticated === 'false'", login_interface_ui("authenticate")
#     ),
#     ui.panel_conditional(
#         "output.authenticated === 'true'",
#         ui.input_slider("n", "Number of bins", min=1, max=20, value=10),
#     ),
#     ui.panel_conditional(
#         "output.authenticated === 'true'",
#         output_widget("rank_response_overview_plot")
#     ),
#     ui.panel_conditional(
#         "output.authenticated === 'true'",
#         ui.div(
#             ui.h2("UpSetJS"),
#             ui.p("This is a test of the UpSetJS widget."),
#             output_widget("upsetjs_plot"),
#         ),
#     ),
#     ui.panel_conditional(
#         "output.authenticated === 'true'",
#         ui.p("Selected items: ", ui.output_text("selected_items")),
#     ),
# )
