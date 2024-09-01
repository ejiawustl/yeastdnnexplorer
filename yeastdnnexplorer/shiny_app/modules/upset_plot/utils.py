import logging

from shiny import reactive, req

from yeastdnnexplorer.interface import BindingAPI, ExpressionAPI


async def get_set_tfs(
    api_instance: BindingAPI | ExpressionAPI,
    tf_list_reactive: reactive.Value[list[str]],
    logger: logging.Logger,
) -> None:
    """
    Set/update the `tf_list_reactive` with the tf_list extracted from the api_instance.
    This is intended to extract and set a list of regulator symbols from the metadata of
    either a BindingAPI or ExpressionAPI instance.

    :param api_instance: The BindingAPI or ExpressionAPI instance to extract the TFs
        from.
    :type api_instance: BindingAPI | ExpressionAPI
    :param tf_list_reactive: The reactive value to set with the extracted TFs.
    :type tf_list_reactive: reactive.Value[List[str]]
    :param logger: The logger to log messages to.
    :type logger: logging.Logger

    """

    @reactive.effect()
    async def _(api_instance=api_instance, tf_list_reactive=tf_list_reactive):
        """Set/update the _binding_assay_options reactive value with the available
        binding assays."""
        req(api_instance)
        logger.info("Fetching binding assay list")
        res = await api_instance.read()
        df = res.get("metadata")
        try:
            tf_list = df.regulator_symbol.unique().tolist()
            tf_list_reactive.set(tf_list)
            logger.debug(
                f"TFs for class {type(api_instance)} "
                f"with params: {str(api_instance.params)}: {tf_list}"
            )
        except AttributeError as exc:
            logger.error(
                f"The 'regulator_symbol' column was not found in the binding metadata. "
                f"Error: {exc}"
            )
