import os
from typing import Any

import pandas as pd

from yeastdnnexplorer.interface.AbstractRecordsOnlyAPI import AbstractRecordsOnlyAPI


class ExpressionManualQCAPI(AbstractRecordsOnlyAPI):
    """A class to interact with the ExpressionManualQCAPI endpoint."""

    def __init__(self, **kwargs):
        """
        Initialize the ExpressionManualQCAPI object.

        :param kwargs: parameters to pass to AbstractAPI via AbstractRecordsOnlyAPI.

        """
        valid_param_keys = kwargs.pop(
            "valid_param_keys",
            [
                "id",
                "expression",
                "strain_verified",
                "regulator_locus_tag",
                "regulator_symbol",
                "batch",
                "replicate",
                "control",
                "mechanism",
                "restriction",
                "time",
                "source",
                "lab",
                "assay",
                "workflow",
            ],
        )

        url = kwargs.pop("url", os.getenv("EXPRESSIONMANUALQC_URL", None))
        if not url:
            raise AttributeError(
                "url must be provided or the environmental variable ",
                "`EXPRESSIONMANUALQC_URL` must be set",
            )

        super().__init__(url=url, valid_keys=valid_param_keys, **kwargs)

    def create(self, data: dict[str, Any], **kwargs) -> Any:
        raise NotImplementedError("The ExpressionManualQCAPI does not support create.")

    def update(self, df: pd.DataFrame, **kwargs) -> Any:
        raise NotImplementedError("The ExpressionManualQCAPI does not support update.")

    def delete(self, id: str, **kwargs) -> Any:
        raise NotImplementedError("The ExpressionManualQCAPI does not support delete.")

    def submit(self, post_dict: dict[str, Any], **kwargs) -> Any:
        raise NotImplementedError("The ExpressionManualQCAPI does not support submit.")

    def retrieve(
        self, group_task_id: str, timeout: int, polling_interval: int, **kwargs
    ) -> Any:
        raise NotImplementedError(
            "The ExpressionManualQCAPI does not support retrieve."
        )
