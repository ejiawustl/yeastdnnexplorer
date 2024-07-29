import logging

import plotly.graph_objects as go
from shiny import Inputs, Outputs, Session, module, reactive, ui
from shinywidgets import output_widget, render_widget

from yeastdnnexplorer.interface import ExpressionAPI, PromoterSetSigAPI, RankResponseAPI
from yeastdnnexplorer.shiny_app.modules.rank_response_plot.utils import (
    retrieve_rank_response_df,
)

logger = logging.getLogger("shiny")


# Define the UI for the module
@module.ui
def rank_response_plot_ui():
    return ui.card(output_widget("rank_response_plot"))


# Define the server logic for the module
@module.server
def rank_response_plot_server(
    input: Inputs,
    output: Outputs,
    session: Session,
):
    _pss = reactive.Value(
        PromoterSetSigAPI(params={"regulator_symbol": "RTG3", "assay": "callingcards"})
    )
    _expression = reactive.Value(
        ExpressionAPI(
            params={
                "regulator_symbol": "RTG3",
                "source_name": "mcisaac_oe",
                "time": "15",
            }
        )
    )
    _rr = reactive.Value(RankResponseAPI())
    _rank_response_dict = reactive.Value()

    @reactive.effect()
    async def _():
        output_dict = await retrieve_rank_response_df(
            pss=_pss.get(), expression=_expression.get(), rr=_rr.get(), logger=logger
        )
        _rank_response_dict.set(output_dict)

    @render_widget()
    def rank_response_plot():
        rr_dict = _rank_response_dict.get()

        logger.info("Creating rank response plot")
        fig = go.Figure()

        for key, rr_tuple in rr_dict.items():
            # metadata = rr_tuple[0]
            data = rr_tuple[1].iloc[0:50, :]

            # Add each line to the figure
            fig.add_trace(
                go.Scatter(
                    x=data["rank_bin"],
                    y=data["response_ratio"],
                    mode="lines",
                    name=key,  # Optionally, use the key or metadata as the name
                )
            )

        # Update the layout of the figure
        fig.update_layout(
            title={"text": "Rank Response", "x": 0.5},
            yaxis_title="# Responsive / # genes",
            xaxis_title="Number of genes, ranked by binding score",
        )

        return fig
