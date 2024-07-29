import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from shiny import Inputs, Outputs, Session, module, reactive, req, ui
from shinywidgets import output_widget, render_widget
from upsetjs_jupyter_widget import UpSetJSWidget, UpSetSetCombination

from yeastdnnexplorer.interface import BindingAPI, ExpressionAPI

logger = logging.getLogger("shiny")

# Create a ThreadPoolExecutor to run database tasks in a separate thread
executor = ThreadPoolExecutor()


# Define the UI for the module
@module.ui
def upset_plot_ui():
    return ui.card(
        output_widget("upsetjs_plot"),
    )


# Define the server logic for the module
@module.server
def upset_plot_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    binding_reactives: dict[str, reactive.Value | dict[str, reactive.Value]],
    expression_reactives: dict[str, reactive.Value | dict[str, reactive.Value]],
    upset_reactives: dict[str, reactive.Value],
):
    _upset_plot_object = reactive.Value()
    _upset_combinations_list = reactive.Value()
    _selected_dataset = reactive.Value()

    # Define the Binding and Expression API instances and corresponding reactive values
    api_instances = {
        "callingcards": BindingAPI(params={"assay": "callingcards"}),
        "chipexo": BindingAPI(params={"assay": "chipexo"}),
        "harbison_chip": BindingAPI(params={"source_name": "harbison_chip"}),
        " mcisaac_oe": ExpressionAPI(params={"source_name": "mcisaac_oe"}),
        " kemmeren_tfko": ExpressionAPI(params={"source_name": "kemmeren_tfko"}),
        " hu_reimann_tfko": ExpressionAPI(params={"source_name": "hu_reimann_tfko"}),
    }
    tf_lists = {key: reactive.Value() for key in api_instances.keys()}

    # Function to fetch TF list from API instance (to be run in executor)
    def fetch_tf_list(api_instance):
        res = asyncio.run(api_instance.read())
        df = res.get("metadata")
        return df.regulator_symbol.unique().tolist()

    # Use ExtendedTask to fetch TF lists asynchronously
    for key, api_instance in api_instances.items():

        @reactive.extended_task
        async def fetch_task(api_instance=api_instance, key=key):
            loop = asyncio.get_event_loop()
            tf_list = await loop.run_in_executor(executor, fetch_tf_list, api_instance)
            tf_lists[key].set(tf_list)
            logger.debug(f"TFs for class {type(api_instance)}: {tf_list[key].get()}")

        # Invoke the task (e.g., on app startup)
        fetch_task()

    # Reactive output to display the UpSetJS plot
    @reactive.effect()
    def _():
        # Create the UpSetJSWidget instance
        w = UpSetJSWidget[str]()

        # Populate the widget with data
        w.from_dict(
            {f"{key}": tf_lists[key].get() for key in tf_lists}, order_by="name"
        )
        w.generate_intersections(
            order_by="degree",
            min_degree=2,  # Minimum 2 sets in an intersection
            empty=True,  # Include empty intersections
        )

        # Set the widget's mode to "click" to enable selection on click
        w.mode = "click"

        # add a title and description
        w.title = "Chart Title"
        w.description = "a long chart description"
        # w.set_name = "Set Label"
        # w.combination_name = "Combination Label"

        #
        # Define a function to capture selection changes
        def selection_changed(s):
            upset_reactives["sets"].set(s.sets if s else None)
            upset_reactives["regulators"].set(s.elems if s else None)

        # Attach the callback to the widget's selection change event
        w.on_selection_changed(selection_changed)

        combinations_list: list[set[UpSetSetCombination]] = [set()] * len(
            w.combinations
        )
        for i, combination in enumerate(w.combinations):
            combinations_list[i] = {x.name.strip() for x in combination.sets}

        _upset_plot_object.set(w)
        _upset_combinations_list.set(combinations_list)

    # Render the UpSetJSWidget
    @render_widget()
    def upsetjs_plot():
        return _upset_plot_object.get()

    # Set the selected dataset
    @reactive.effect()
    def _():
        binding_assays = binding_reactives["assay"].get()  # type: ignore
        expression_assays = expression_reactives["assay"].get()  # type: ignore

        # Initialize an empty set for selected data sets
        data_sets = set()

        # Check for binding assays
        if "callingcards" in binding_assays:
            data_sets.add("callingcards")
        if "chipexo" in binding_assays:
            data_sets.add("chipexo")
        if "chip" in binding_assays:
            data_sets.add("harbison_chip")

        # Check for tfko in expression assays before accessing its source
        if "tfko" in expression_assays:
            tfko_source = expression_reactives["tfko"]["source"].get()
            if "kemmeren" in tfko_source:
                data_sets.add("kemmeren_tfko")
            if "hu_reimann" in tfko_source:
                data_sets.add("hu_reimann_tfko")

        # Check for overexpression in expression assays
        if "overexpression" in expression_assays:
            data_sets.add("mcisaac_oe")

        logger.info(f"Data sets: {data_sets}")

        # Update the selected dataset reactive
        _selected_dataset.set(data_sets)

    # React to changes in binding and expression assay selections
    @reactive.effect()
    @reactive.event(_selected_dataset)
    def _():
        req(_upset_plot_object)
        req(_upset_combinations_list)

        w = _upset_plot_object.get()
        w_new = w.copy()

        # if multiple assays are selected from binding and/or expression, then find
        # the index of the combination that matches and set that as the selection
        # in the upset plot object.
        if len(_selected_dataset.get()) < 2:
            if w_new.selection:
                w_new.selection = None
                _upset_plot_object.set(w_new)
        else:
            for i, combination in enumerate(_upset_combinations_list.get()):
                if _selected_dataset.get() == combination:
                    # w_new.selection = w.combinations[i]
                    w_new.selection = w.combinations[i]
                    logger.debug(
                        f"Setting selection in upset plot object: {w_new.selection}"
                    )
                    _upset_plot_object.set(w_new)
                    break

    # Register the shutdown callback
    session.on_ended(on_shutdown)


# Note: Make sure to shut down the executor when the app stops
def on_shutdown():
    logger.debug("Shutting down executor")
    executor.shutdown()
