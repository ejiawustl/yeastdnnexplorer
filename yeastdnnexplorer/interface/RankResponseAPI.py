import asyncio
import json
import os
import tarfile
import tempfile
import time
from collections.abc import Callable
from typing import Any

import aiohttp
import pandas as pd

from yeastdnnexplorer.interface.AbstractRecordsAndFilesAPI import (
    AbstractRecordsAndFilesAPI,
)


class RankResponseAPI(AbstractRecordsAndFilesAPI):
    """
    A class to interact with the Rank Response API.

    Retrieves rank response data from the database.

    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize the RankResponseAPI object. This will serve as an interface to the
        RankResponse endpoint of both the database and the application cache.

        :param url: The URL of the Rank Response API
        :param kwargs: Additional parameters to pass to AbstractAPI.

        """
        super().__init__(
            url=kwargs.pop("url", os.getenv("PROMOTERSETSIG_URL", "")),
            export_files_url_suffix="rankresponse",
            **kwargs,
        )

    async def submit(
        self,
        post_dict: dict[str, Any],
        **kwargs,
    ) -> Any:
        # make a post request with the post_dict to rankresponse_url
        rankresponse_url = f"{self.url.rstrip('/')}/rankresponse/"
        self.logger.debug("rankresponse_url: %s", rankresponse_url)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                rankresponse_url, headers=self.header, json=post_dict
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
        :return: Extracted files from the result tarball.

        """
        # Start time for timeout check
        start_time = time.time()

        # Task status URL
        status_url = f"{self.url.rstrip('/')}/rankresponse_task_status/"

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
                        # Fetch and return the tarball
                        return await self._download_result(group_task_id)
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

    async def _download_result(self, group_task_id: str) -> Any:
        """
        Download the result tarball after the task is successful.

        :param group_task_id: The group_task_id to download the results for.
        :return: Extracted metadata and data from the tarball.

        """
        download_url = f"{self.url.rstrip('/')}/rankresponse_get_data/"

        async with aiohttp.ClientSession() as session:
            async with session.get(
                download_url,
                headers=self.header,
                params={"group_task_id": group_task_id},
            ) as response:
                response.raise_for_status()  # Ensure request was successful
                tar_data = await response.read()

                # Save tarball to a temporary file or return raw tar content
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".tar.gz"
                ) as temp_file:
                    temp_file.write(tar_data)
                    temp_file.flush()
                    temp_file.seek(0)

                    # Extract and return the content of the tarball
                    return self._extract_files(temp_file.name)

    def _extract_files(self, tar_path: str) -> dict[str, pd.DataFrame]:
        """
        Extract metadata and associated files from a tarball.

        :param tar_path: The path to the tarball file.
        :return: A tuple of metadata DataFrame and a dictionary of DataFrames for each
            file.

        """
        with tarfile.open(tar_path, mode="r:gz") as tar:
            tar_members = tar.getmembers()

            # Extract metadata.json
            metadata_member = next(
                (m for m in tar_members if m.name == "metadata.json"), None
            )
            if metadata_member is None:
                raise FileNotFoundError("metadata.json not found in tar archive")

            extracted_file = tar.extractfile(metadata_member)
            if extracted_file is None:
                raise FileNotFoundError("Failed to extract metadata.json")

            with extracted_file as f:
                metadata_dict = json.load(f)

            metadata_df = pd.DataFrame(metadata_dict.values())
            metadata_df["id"] = metadata_dict.keys()

            # Extract CSV files
            data = {}
            for rr_id in metadata_df["id"]:
                csv_filename = f"{rr_id}.csv.gz"
                member = next((m for m in tar_members if m.name == csv_filename), None)
                if member is None:
                    raise FileNotFoundError(f"{csv_filename} not found in tar archive")

                extracted_file = tar.extractfile(member)
                if extracted_file is None:
                    raise FileNotFoundError(f"Failed to extract {csv_filename}")

                with extracted_file as f:
                    data[rr_id] = pd.read_csv(f, compression="gzip")
        return {"metadata": metadata_df, "data": data}

    async def read(
        self,
        callback: Callable[
            [pd.DataFrame, dict[str, Any] | None, Any], Any
        ] = lambda metadata, data, cache, **kwargs: (
            {"metadata": metadata, "data": data}
        ),
        retrieve_files: bool = False,
        **kwargs,
    ) -> Any:
        raise NotImplementedError("The RankResponseAPI does not support read.")

    def create(self, data: dict[str, Any], **kwargs) -> Any:
        raise NotImplementedError("The RankResponseAPI does not support create.")

    def update(self, df: pd.DataFrame, **kwargs) -> Any:
        raise NotImplementedError("The RankResponseAPI does not support update.")

    def delete(self, id: str, **kwargs) -> Any:
        raise NotImplementedError("The RankResponseAPI does not support delete.")
