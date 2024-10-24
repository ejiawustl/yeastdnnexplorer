import logging

import plotly.graph_objects as go
from shiny import Inputs, Outputs, Session, module, reactive, ui
from shinywidgets import output_widget, render_widget

from yeastdnnexplorer.interface import ExpressionAPI, PromoterSetSigAPI, RankResponseAPI
from yeastdnnexplorer.shiny_app.modules.rank_response_plot.utils import (
    binom_ci,
    submit_rank_response_job,
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
    _rank_response_keys = reactive.Value()

    @reactive.effect()
    async def _():
        group_id = await submit_rank_response_job(
            pss=_pss.get(), expression=_expression.get(), rr=_rr.get(), logger=logger
        )
        _rank_response_keys.set(group_id)

    @render_widget()
    async def rank_response_plot():
        rr_dict = await _rr.get().retrieve(_rank_response_keys.get())

        logger.info("Creating rank response plot")
        fig = go.Figure()

        random_table = rr_dict.get("metadata").loc[0, "id"]

        for _, row in rr_dict.get("metadata").iterrows():
            data = rr_dict.get("data").get(row["id"]).iloc[0:50, :]

            key = row["promotersetsig_ids"] + "_" + row["expression_ids"]

            # Add each line to the figure
            fig.add_trace(
                go.Scatter(
                    x=data["rank_bin"],
                    y=data["response_ratio"],
                    mode="lines",
                    name=key,  # Optionally, use the key or metadata as the name
                )
            )
            if row["id"] == random_table:

                ci_lower_upper = data["rank_bin"].apply(
                    lambda n: binom_ci(n, data["random"][0])
                )
                data = data.assign(
                    ci_lower=ci_lower_upper.apply(lambda x: x[0]),
                    ci_upper=ci_lower_upper.apply(lambda x: x[1]),
                )

                # Add the random line with a black dashed style
                fig.add_trace(
                    go.Scatter(
                        x=data["rank_bin"],
                        y=data["random"],
                        mode="lines",
                        name="Random",
                        line=dict(dash="dash", color="black"),
                    )
                )

                # Add the confidence interval lower bound trace
                fig.add_trace(
                    go.Scatter(
                        x=data["rank_bin"],
                        y=data["ci_lower"],
                        mode="lines",
                        line=dict(width=0),  # No visible line
                        showlegend=False,
                    )
                )

                # Add the confidence interval upper bound trace and shade
                # the area (tonexty)
                fig.add_trace(
                    go.Scatter(
                        x=data["rank_bin"],
                        y=data["ci_upper"],
                        mode="lines",
                        fill="tonexty",
                        fillcolor="rgba(128, 128, 128, 0.3)",
                        line=dict(width=0),
                        showlegend=False,
                    )
                )

        # Update the layout of the figure
        fig.update_layout(
            title={"text": "Rank Response", "x": 0.5},
            yaxis_title="# Responsive / # genes",
            xaxis_title="Number of genes, ranked by binding score",
        )

        return fig
