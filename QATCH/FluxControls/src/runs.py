import json
import requests
from typing import Union
try:
    from src.constants import Actions, Lights, HEADERS

    class Log:
        def d(tag, msg=""): print("DEBUG:", tag, msg)
        def i(tag, msg=""): print("INFO:", tag, msg)
        def w(tag, msg=""): print("WARNING:", tag, msg)
        def e(tag, msg=""): print("ERROR:", tag, msg)
    Log.i(print("Running FluxControls as standalone app"))

except (ImportError, ModuleNotFoundError):
    from QATCH.common.logger import Logger as Log
    from QATCH.FluxControls.src.constants import Actions, Lights, HEADERS


class Run:
    def __init__(self):
        pass


class Runs:
    """
    A class that provides methods to interact with the Opentrons Flex HTTP API for managing protocols, runs, and robotic actions.

    This class includes static methods to perform common operations related to Opentrons Flex systems, including:
    - Managing protocols (uploading, deleting, listing).
    - Controlling the state of runs (starting, stopping, pausing, playing).
    - Fetching status and light control for Flex systems.
    - Sending homing commands to the Flex robot.

    Each method uses the `_send_request` utility to make HTTP requests (GET, POST, DELETE) to the Flex API and handles
    responses accordingly. Methods are designed to be flexible and provide easy access to different types of Flex operations
    based on the given URLs and parameters.

    Methods:
        - `upload_protocol`: Uploads a protocol file to the Flex system.
        - `upload_protocol_custom_labware`: Uploads a protocol and custom labware files to the Flex system.
        - `delete_protocol`: Deletes a specified protocol from the Flex system.
        - `get_protocols_list`: Retrieves a list of available protocols.
        - `get_run_status`: Fetches the status of a specific run.
        - `get_runs_list`: Retrieves a list of runs.
        - `pause_run`: Pauses an ongoing run.
        - `play_run`: Starts or resumes a run.
        - `stop_run`: Stops a running protocol.
        - `resume_run`: Resumes a running protocol
        - `set_lights`: Sets the status of the Flex system's lights.
        - `get_lights`: Fetches the current light status.
        - `home`: Sends a homing command to the Flex robot to return to its reference position.

    Attributes:
        None.

    This class allows users to easily control and monitor their Opentrons Flex by encapsulating the interactions with the API.
    """

    @staticmethod
    def _send_request(method: str, url: str, payload: dict = None) -> Union[dict, None]:
        """
        Sends an HTTP request to the specified URL using the specified method.

        Supports `POST`, `GET`, and `DELETE` methods. For `POST` requests, handles payloads
        containing JSON data or file uploads.

        Args:
            method (str): The HTTP method to use. Must be one of "POST", "GET", or "DELETE".
            url (str): The URL to which the request will be sent.
            payload (dict, optional): The payload to include in the request. For `POST`:
                - If `payload` is a dictionary containing the key "files", it is treated as a file upload.
                - If `payload` is a list of tuples, where each tuple starts with "files", it is treated as a file upload.
                - Otherwise, the payload is serialized to JSON format.

        Logs:
            Logs the errors that occur when sending requests and recieving responses from Flex.

        Returns:
            dict: The JSON response from the server if the request is successful.
            None: If the request fails or encounters an exception.

        Raises:
            ValueError: If the provided HTTP method is unsupported.
            requests.exceptions.RequestException: If the HTTP request fails due to a network issue or server error.
        """
        try:

            if method == "POST":
                if payload:
                    headers = {"Content-Type": "application/json"}
                    headers.update(HEADERS)
                    request_kwargs = {"url": url, "headers": headers}
                    if isinstance(payload, dict) and "files" in payload:
                        request_kwargs["files"] = payload
                    elif isinstance(payload, list) and all(
                        isinstance(item, tuple) and item[0] == "files"
                        for item in payload
                    ):
                        request_kwargs["files"] = payload
                    else:
                        request_kwargs["data"] = json.dumps(payload)
                response = requests.post(**request_kwargs)
            elif method == "GET":
                request_kwargs = {"url": url, "headers": HEADERS}
                response = requests.get(**request_kwargs)
            elif method == "DELETE":
                request_kwargs = {"url": url, "headers": HEADERS}
                response = requests.delete(**request_kwargs)
            else:
                Log.e(f"Unsupported HTTP method: {method}")
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            Log.e(f"Error sending {method} request to {url}: {e}")
            return None

    @staticmethod
    def run_protocol(runs_url: str, protocol_id: str) -> str:
        """
        Executes a protocol by sending a POST request to the specified runs URL.

        This method sends a request with the provided `protocol_id` to start a run.
        The `protocolId` is included in the payload.

        Args:
            runs_url (str): The URL to which the POST request will be sent to initiate the protocol run.
            protocol_id (str): The unique identifier of the protocol to be executed.

        Logs:
            Logs the POST operation with the target URL for debugging and informational purposes.

        Returns:
            str: The ID of the initiated run, extracted from the server's response.

        Raises:
            KeyError: If the response does not contain the expected "data" or "id" fields.
            requests.exceptions.RequestException: If the HTTP request fails (handled internally by `_send_request`).
        """
        Log.i(f"POST: run protocol to {runs_url}")
        payload = {"data": {"protocolId": protocol_id}}
        response = Runs._send_request("POST", runs_url, payload)
        return response["data"]["id"]

    @staticmethod
    def delete_protocol(protocols_url: str, protocol_id: str) -> Union[dict, None]:
        """
        Deletes a protocol by sending a DELETE request to the specified protocols URL.

        Constructs a URL by appending the `protocol_id` to the `protocols_url` and
        sends a DELETE request to remove the specified protocol.

        Args:
            protocols_url (str): The base URL for protocols.
            protocol_id (str): The unique identifier of the protocol to be deleted.

        Returns:
            dict: The JSON response from the server if the request is successful.
            None: If the request fails or encounters an exception (handled internally by `_send_request`).

        Logs:
            Logs the DELETE operation with the target URL for debugging and informational purposes.

        Raises:
            requests.exceptions.RequestException: If the HTTP request fails (handled internally by `_send_request`).
        """
        Log.i(f"DELETE: remove protocol from {protocols_url}")
        delete_protocol_url = f"{protocols_url}/{protocol_id}"
        return Runs._send_request("DELETE", delete_protocol_url)

    @staticmethod
    def upload_protocol(protocols_url: str, protocol_file_path: str) -> dict:
        """
        Uploads a protocol file to the specified protocols URL.

        Reads the protocol file from the provided file path, sends a POST request to upload
        it, and returns the protocol's ID and name from the server's response.

        Args:
            protocols_url (str): The URL to which the protocol file will be uploaded.
            protocol_file_path (str): The local file path of the protocol file to be uploaded.

        Returns:
            dict: A dictionary containing the following keys:
                - `protocol_id` (str): The unique identifier of the uploaded protocol.
                - `protocol_name` (str): The name of the uploaded protocol, extracted from the metadata.

        Logs:
            Logs the file upload operation, including the file path and target URL.

        Raises:
            KeyError: If the response does not contain the expected "data", "id", or "metadata" fields.
            FileNotFoundError: If the specified protocol file cannot be found or opened.
            requests.exceptions.RequestException: If the HTTP request fails (handled internally by `_send_request`).
        """
        Log.i(
            f"POST: uploading protocol file from path {protocol_file_path} to {protocols_url}"
        )
        protocol_file_payload = open(protocol_file_path, "rb")
        data = {"files": protocol_file_payload}
        response = Runs._send_request("POST", protocols_url, data)
        protocol_file_payload.close()
        return {
            "protocol_id": response["data"]["id"],
            "protocol_name": response["data"]["metadata"]["protocolName"],
        }

    @staticmethod
    def upload_protocol_custom_labware(
        protocols_url: str, protocol_file_path: str, labware_file_paths: list[str]
    ) -> dict:
        """
        Uploads a protocol file along with associated custom labware files to the specified URL.

        Reads the protocol file and any custom labware files from the provided file paths, sends
        a POST request to upload them, and returns the protocol's ID and name from the server's response.

        Args:
            protocols_url (str): The URL to which the protocol and labware files will be uploaded.
            protocol_file_path (str): The local file path of the protocol file to be uploaded.
            labware_file_paths (list[str]): A list of file paths for the custom labware files to be uploaded.

        Returns:
            dict: A dictionary containing the following keys:
                - `protocol_id` (str): The unique identifier of the uploaded protocol.
                - `protocol_name` (str): The name of the uploaded protocol, extracted from the metadata.

        Logs:
            Logs the upload operation, including the file paths and target URL.

        Raises:
            KeyError: If the response does not contain the expected "data", "id", or "metadata" fields.
            FileNotFoundError: If any of the specified files cannot be found or opened.
            requests.exceptions.RequestException: If the HTTP request fails (handled internally by `_send_request`).

        Notes:
            Ensures that all opened file handles are properly closed, even in the event of an error.
        """
        Log.i(
            f"POST: uploading custom labware protocol file from path {protocol_file_path} and {labware_file_paths} to {protocols_url}"
        )
        protocol_file_payload = open(protocol_file_path, "rb")
        data = [("files", protocol_file_payload)]

        labware_file_payloads = []
        try:
            for labware_file_path in labware_file_paths:
                labware_payload = open(labware_file_path, "rb")
                labware_file_payloads.append(labware_payload)
                data.append(("files", labware_payload))

            response = Runs._send_request("POST", protocols_url, data)
            return {
                "protocol_id": response["data"]["id"],
                "protocol_name": response["data"]["metadata"]["protocolName"],
            }
        finally:
            protocol_file_payload.close()
            for labware_payload in labware_file_payloads:
                labware_payload.close()

    @staticmethod
    def delete_run(runs_url: str, run_id: int) -> Union[dict, None]:
        """
        Deletes a specific run by sending a DELETE request to the specified runs URL.

        Constructs a URL by appending the `run_id` to the `runs_url` and sends a DELETE
        request to remove the specified run.

        Args:
            runs_url (str): The base URL for the runs.
            run_id (int): The unique identifier of the run to be deleted.

        Returns:
            dict: The JSON response from the server if the request is successful.
            None: If the request fails or encounters an exception (handled internally by `_send_request`).

        Logs:
            Logs the DELETE operation, including the target URL and run ID, for debugging and informational purposes.

        Raises:
            requests.exceptions.RequestException: If the HTTP request fails (handled internally by `_send_request`).
        """
        Log.i(f"DELETE: removing run from {runs_url} with ID {run_id}")
        delete_run_url = f"{runs_url}/{run_id}"
        return Runs._send_request("DELETE", delete_run_url)

    @staticmethod
    def get_protocols_list(protocols_url: str) -> list:
        """
        Retrieves the list of protocols from the specified URL.

        Sends a GET request to the provided `protocols_url` and parses the response to
        extract the list of protocols.

        Args:
            protocols_url (str): The URL from which to fetch the protocols list.

        Returns:
            list: A list of protocols, where each protocol is represented as a dictionary containing its details.

        Logs:
            Logs the GET operation, including the target URL, for debugging and informational purposes.

        Raises:
            KeyError: If the response does not contain the expected "data" field.
            requests.exceptions.RequestException: If the HTTP request fails (handled internally by `_send_request`).
        """
        Log.i(f"GET: fetching protocols list from {protocols_url}")
        response = Runs._send_request("GET", protocols_url)
        return [protocol for protocol in response["data"]]

    @staticmethod
    def create_run(runs_url: str, protocol_id: str):
        Log.i(
            f"POST: creating run from {runs_url} from protocol id {protocol_id}")
        payload = {"data": {"protocolId": f"{protocol_id}"}}
        return Runs._send_request("POST", runs_url, payload)

    @staticmethod
    def get_run_status(runs_url: str, run_id: int) -> dict:
        """
        Retrieves the status of a specific run by sending a GET request to the specified URL.

        Constructs a URL by appending the `run_id` to the `runs_url` and sends a GET
        request to fetch the status of the specified run.

        Args:
            runs_url (str): The base URL for the runs.
            run_id (int): The unique identifier of the run for which the status is requested.

        Returns:
            dict: A dictionary containing the run status and related information, as returned by the server.

        Logs:
            Logs the GET operation, including the target URL and run ID, for debugging and informational purposes.

        Raises:
            requests.exceptions.RequestException: If the HTTP request fails (handled internally by `_send_request`).
        """
        Log.i(
            f"GET: fetching run status from {runs_url} with ID {run_id}")
        status_url = f"{runs_url}/{run_id}"
        return Runs._send_request("GET", status_url)

    @staticmethod
    def get_runs_list(runs_url: str) -> list:
        """
        Retrieves the list of runs from the specified URL.

        Sends a GET request to the provided `runs_url` and parses the response to extract
        the list of runs.

        Args:
            runs_url (str): The URL from which to fetch the list of runs.

        Returns:
            list: A list of runs, where each run is represented as a dictionary containing its details.

        Logs:
            Logs the GET operation, including the target URL, for debugging and informational purposes.

        Raises:
            KeyError: If the response does not contain the expected "data" field.
            requests.exceptions.RequestException: If the HTTP request fails (handled internally by `_send_request`).
        """
        Log.i(f"GET: fetching run list from {runs_url}")
        response = Runs._send_request("GET", runs_url)
        return [run for run in response["data"]]

    @staticmethod
    def pause_run(runs_url: str, run_id: str) -> dict:
        """
        Pauses a specific run by sending a POST request to the specified URL.

        Constructs an action URL by appending the `run_id` to the `runs_url` and sends a
        POST request to trigger the pause action for the specified run.

        Args:
            runs_url (str): The base URL for the runs.
            run_id (str): The unique identifier of the run to be paused.

        Returns:
            dict: The JSON response from the server if the request is successful, indicating the result of the pause action.

        Logs:
            Logs the POST operation, including the target URL and run ID, for debugging and informational purposes.

        Raises:
            requests.exceptions.RequestException: If the HTTP request fails (handled internally by `_send_request`).
        """
        Log.i(f"POST: pausing run from {runs_url} with ID {run_id}")
        actions_url = f"{runs_url}/{run_id}/actions"
        action_payload = {"data": {"actionType": Actions.PAUSE.value}}
        return Runs._send_request("POST", actions_url, action_payload)

    @staticmethod
    def play_run(runs_url: str, run_id: str) -> dict:
        """
        Resumes a specific run by sending a POST request to the specified URL.

        Constructs an action URL by appending the `run_id` to the `runs_url` and sends a
        POST request to trigger the play (resume) action for the specified run.

        Args:
            runs_url (str): The base URL for the runs.
            run_id (str): The unique identifier of the run to be resumed.

        Returns:
            dict: The JSON response from the server if the request is successful, indicating the result of the play action.

        Logs:
            Logs the POST operation, including the target URL and run ID, for debugging and informational purposes.

        Raises:
            requests.exceptions.RequestException: If the HTTP request fails (handled internally by `_send_request`).
        """
        Log.i(f"POST: playing run from {runs_url} with ID {run_id}")
        actions_url = f"{runs_url}/{run_id}/actions"
        action_payload = {"data": {"actionType": Actions.PLAY.value}}
        return Runs._send_request("POST", actions_url, action_payload)

    @staticmethod
    def stop_run(runs_url: str, run_id: str) -> dict:
        """
        Stops a specific run by sending a POST request to the specified URL.

        Constructs an action URL by appending the `run_id` to the `runs_url` and sends a
        POST request to trigger the stop action for the specified run.

        Args:
            runs_url (str): The base URL for the runs.
            run_id (str): The unique identifier of the run to be stopped.

        Returns:
            dict: The JSON response from the server if the request is successful, indicating the result of the stop action.

        Logs:
            Logs the POST operation, including the target URL and run ID, for debugging and informational purposes.

        Raises:
            requests.exceptions.RequestException: If the HTTP request fails (handled internally by `_send_request`).
        """
        Log.i(f"POST: stopping run from {runs_url} with ID {run_id}")
        actions_url = f"{runs_url}/{run_id}/actions"
        action_payload = {"data": {"actionType": Actions.STOP.value}}
        return Runs._send_request("POST", actions_url, action_payload)

    @staticmethod
    def resume_run(runs_url: str, run_id: str) -> dict:
        """
        Resumes a specific run by sending a POST request to the specified URL.

        Constructs an action URL by appending the `run_id` to the `runs_url` and sends a
        POST request to trigger the stop action for the specified run.

        Args:
            runs_url (str): The base URL for the runs.
            run_id (str): The unique identifier of the run to be stopped.

        Returns:
            dict: The JSON response from the server if the request is successful, indicating the result of the resume action.

        Logs:
            Logs the POST operation, including the target URL and run ID, for debugging and informational purposes.

        Raises:
            requests.exceptions.RequestException: If the HTTP request fails (handled internally by `_send_request`).
        """
        Log.i(f"POST: resuming run from {runs_url} with ID {run_id}")
        actions_url = f"{runs_url}/{run_id}/actions"
        action_payload = {"data": {"actionType": Actions.RESUME.value}}
        return Runs._send_request("POST", actions_url, action_payload)

    @staticmethod
    def set_lights(lights_url: str, light_status: Lights) -> dict:
        """
        Sets the status of the lights by sending a POST request to the specified URL.

        Sends a POST request to the `lights_url` with the specified `light_status` to
        control the state of the lights (e.g., ON, OFF).

        Args:
            lights_url (str): The URL where the light control request is sent.
            light_status (FlexLights): The desired status of the lights (e.g., ON or OFF).

        Returns:
            dict: The JSON response from the server indicating the result of the light status change.

        Logs:
            Logs the POST operation, including the target URL and light status, for debugging and informational purposes.

        Raises:
            requests.exceptions.RequestException: If the HTTP request fails (handled internally by `_send_request`).
        """
        Log.i(f"POST: setting lights to {light_status} at {lights_url}")
        return Runs._send_request("POST", lights_url, light_status)

    @staticmethod
    def get_lights(lights_url: str) -> dict:
        """
        Fetches the current status of the lights by sending a GET request to the specified URL.

        Sends a GET request to the `lights_url` to retrieve the current state of the lights.

        Args:
            lights_url (str): The URL where the light status request is sent.

        Returns:
            dict: The JSON response from the server containing the current light status.

        Logs:
            Logs the GET operation, including the target URL, for debugging and informational purposes.

        Raises:
            requests.exceptions.RequestException: If the HTTP request fails (handled internally by `_send_request`).
        """
        Log.i(f"GET: fetching light status from {lights_url}")
        return Runs._send_request(method="GET", url=lights_url)

    # @staticmethod
    # def home(home_url: str) -> dict:
    #     """
    #     Homes the Flex robot by sending a POST request to the specified URL.

    #     Sends a POST request to the `home_url` with a payload to command the Flex robot to home (move to a known reference position).

    #     Args:
    #         home_url (str): The URL where the homing request is sent.

    #     Returns:
    #         dict: The JSON response from the server indicating the result of the homing action.

    #     Logs:
    #         Logs the POST operation, including the target URL, for debugging and informational purposes.

    #     Raises:
    #         requests.exceptions.RequestException: If the HTTP request fails (handled internally by `_send_request`).
    #     """
    #     Log.i(f"POST: homing Flex at {home_url}")
    #     command_dict = {"target": "robot"}
    #     command_payload = json.dumps(command_dict)
    #     return Runs._send_request(
    #         method="POST", url=home_url, payload=command_payload
    #     )
