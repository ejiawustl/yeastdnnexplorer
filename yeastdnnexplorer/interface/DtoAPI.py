import asyncio
import json
import os
import time
from typing import Any

import aiohttp
import pandas as pd
from requests import Response  # type: ignore

from yeastdnnexplorer.interface.AbstractRecordsOnlyAPI import AbstractRecordsOnlyAPI


class DtoAPI(AbstractRecordsOnlyAPI):
    """
    A class to interact with the DTO API.

    Retrieves dto data from the database.

    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize the DTO object. This will serve as an interface to the DTO endpoint
        of both the database and the application cache.

        :param url: The URL of the DTO API
        :param kwargs: Additional parameters to pass to AbstractAPI.

        """
        super().__init__(
            url=kwargs.pop("url", os.getenv("DTO_URL", "")),
            **kwargs,
        )

    async def submit(
        self,
        post_dict: dict[str, Any],
        **kwargs,
    ) -> Any:
        # make a post request with the post_dict to dto_url
        dto_url = f"{self.url.rstrip('/')}/submit/"
        self.logger.debug("dto_url: %s", dto_url)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                dto_url, headers=self.header, json=post_dict
            ) as response:
                response.raise_for_status()
                result = await response.json()
                try:
                    return result["group_task_id"]
                except KeyError:
                    self.logger.error(
                        "Expected 'group_task_id' in response: %s", json.dumps(result)
                    )
                    raise

    async def retrieve(
        self,
        group_task_id: str,
        timeout: int = 300,
        polling_interval: int = 2,
        **kwargs,
    ) -> dict[str, pd.DataFrame]:
        """
        Periodically check the task status and retrieve the result when the task
        completes.

        :param group_task_id: The task ID to retrieve results for.
        :param timeout: The maximum time to wait for the task to complete (in seconds).
        :param polling_interval: The time to wait between status checks (in seconds).
        :return: Records from the DTO API of the successfully completed task.

        """
        # Start time for timeout check
        start_time = time.time()

        # Task status URL
        status_url = f"{self.url.rstrip('/')}/status/"

        while True:
            async with aiohttp.ClientSession() as session:
                # Send a GET request to check the task status
                async with session.get(
                    status_url,
                    headers=self.header,
                    params={"group_task_id": group_task_id},
                ) as response:
                    response.raise_for_status()  # Raise an error for bad status codes
                    status_response = await response.json()

                    # Check if the task is complete
                    if status_response.get("status") == "SUCCESS":

                        if error_tasks := status_response.get("error_tasks"):
                            self.logger.error(
                                f"Tasks {group_task_id} failed: {error_tasks}"
                            )
                        if success_tasks := status_response.get("success_tasks"):
                            params = {"id": ",".join(success_tasks)}
                            return await self.read(params=params)
                    elif status_response.get("status") == "FAILURE":
                        raise Exception(
                            f"Task {group_task_id} failed: {status_response}"
                        )

                    # Check if we have reached the timeout
                    elapsed_time = time.time() - start_time
                    if elapsed_time > timeout:
                        raise TimeoutError(
                            f"Task {group_task_id} did not "
                            "complete within {timeout} seconds."
                        )

                    # Wait for the specified polling interval before checking again
                    await asyncio.sleep(polling_interval)

    def create(self, data: dict[str, Any], **kwargs) -> Response:
        raise NotImplementedError("The DTO does not support create.")

    def update(self, df: pd.DataFrame, **kwargs) -> Any:
        raise NotImplementedError("The DTO does not support update.")

    def delete(self, id: str, **kwargs) -> Any:
        raise NotImplementedError("The DTO does not support delete.")
