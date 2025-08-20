import os
import re
from typing import Union
import time
from datetime import datetime
import subprocess

try:
    from src.pipette import Pipette
    from src.constants import (
        MountPositions,
        Pipettes,
        DeckLocations,
        Axis,
        Lights,
        HTTP_PORT,
    )
    from src.commands import Commands
    from src.runs import Runs
    from src.labware import Labware
    from src.standard_labware import StandardLabware

    class Log:
        def d(tag, msg=""):
            print("DEBUG:", tag, msg)

        def i(tag, msg=""):
            print("INFO:", tag, msg)

        def w(tag, msg=""):
            print("WARNING:", tag, msg)

        def e(tag, msg=""):
            print("ERROR:", tag, msg)

    Log.i(print("Running FluxControls as standalone app"))
except (ImportError, ModuleNotFoundError):
    from QATCH.FluxControls.src.pipette import Pipette
    from QATCH.FluxControls.src.constants import (
        MountPositions,
        Pipettes,
        DeckLocations,
        Axis,
        Lights,
        HTTP_PORT,
    )
    from QATCH.FluxControls.src.commands import Commands
    from QATCH.FluxControls.src.runs import Runs
    from QATCH.FluxControls.src.labware import Labware
    from QATCH.FluxControls.src.standard_labware import StandardLabware
    from QATCH.common.logger import Logger as Log


class OpentronsFlex:
    """
    A class representing an Opentrons Flex system for automating laboratory protocols.

    This class provides methods for managing and executing laboratory runs, including
    creating, pausing, stopping, and deleting runs. It also offers functionality to
    validate configurations, control lighting, and manage labware and pipettes.

    Attributes:
        available_protocols (dict): A dictionary of available protocols and their metadata.
        available_labware (dict): A dictionary of available labware for use in the system.
        gantry (dict): A dictionary representing the gantry configuration for pipettes.
        light_state (str): The current state of the system's lights (e.g., 'on' or 'off').

    The OpentronsFlex class interfaces with the FlexRuns and FlexLights systems, allowing
    users to automate lab tasks with precision and flexibility.
    """

    def __init__(self, mac_address: str, ip_address: str = None) -> None:
        """
        Initializes the `OpentronsFlex` instance with the given MAC address and optionally an IP address.

        Args:
            mac_address (str): The MAC address of the Opentrons Flex robot.
            ip_address (str, optional): The IP address of the robot. Defaults to None.

        Returns:
            None
        """
        self._set_robot_mac_address(mac_address)

        self._setup(ip=ip_address)

    def _setup(self, ip: str = None) -> None:
        """
        Configures the Opentrons Flex robot with the provided IP address.

        This method initializes various robot properties, sets up network-related configurations,
        and prepares the robot for operation by defining URLs for runs, protocols, lights, and home.
        It also initializes the available protocols, gantry mounts, and labware locations.

        Args:
            ip (str, optional): The IP address to assign to the robot. Defaults to None.

        Returns:
            None
        """
        Log.i(
            f"Initializing OpentronsFlex with MAC: {self._get_robot_mac_address()}")
        self.available_protocols = {}
        self.gantry = {
            MountPositions.LEFT_MOUNT: None,
            MountPositions.RIGHT_MOUNT: None,
        }

        if ip is None:
            ip = self.find_ip()

        self._set_robot_ipv4(ip)
        Log.i(f"Running flex at IPv4: {self._get_robot_ipv4()}:{HTTP_PORT}")
        self._set_base_url(f"http://{self._get_robot_ipv4()}:{HTTP_PORT}")
        self._set_runs_url(f"{self._get_base_url()}/runs")
        self._set_protocols_url(f"{self._get_base_url()}/protocols")
        self.update_available_protocols()
        self._set_lights_url(f"{self._get_base_url()}/robot/lights")
        self._set_home_url(
            f"{self._get_base_url()}/commands?waitUntilComplete=true")
        self._set_command_url(
            f"{self._get_base_url()}/robot/{self.get_run_list()[0].get('id')}/commands"
        )
        self.lights_on()
        self.available_labware = {
            DeckLocations.A1: None,
            DeckLocations.A2: None,
            DeckLocations.A3: None,
            DeckLocations.A4: None,
            DeckLocations.B1: None,
            DeckLocations.B2: None,
            DeckLocations.B3: None,
            DeckLocations.B4: None,
            DeckLocations.C1: None,
            DeckLocations.C2: None,
            DeckLocations.C3: None,
            DeckLocations.C4: None,
            DeckLocations.D1: None,
            DeckLocations.D2: None,
            DeckLocations.D3: None,
            DeckLocations.D4: None,
        }
        Log.d(f"Setup complete. Base URL: {self._get_base_url()}")

    def load_pipette(self, pipette: Pipettes, position: MountPositions) -> None:
        """
        Loads a pipette onto the specified mount position of the Opentrons Flex robot.

        This method initializes a new pipette, associates it with the specified mount position
        (left or right), and sends the appropriate command to the robot to load the pipette.
        Once the pipette is loaded, its unique ID is retrieved and stored.

        Args:
            pipette (FlexPipettes): The type of pipette to be loaded.
            position (FlexMountPositions): The mount position where the pipette will be loaded
                (LEFT_MOUNT or RIGHT_MOUNT).

        Returns:
            None
        """
        Log.i(
            f"Loading pipette: {pipette.value} at position: {position.value}")
        new_pipette = Pipette(pipette=pipette, mount_position=position)

        if position is MountPositions.LEFT_MOUNT:
            self._set_left_pipette(new_pipette)
            payload = Commands.load_pipette(self._left_pipette)
            response = Commands.send_command(
                command_url=self._get_command_url(), command_dict=payload
            )
            pipette_id = response["data"]["result"]["pipetteId"]
            new_pipette.id = pipette_id
            Log.d(f"Left pipette loaded with ID: {pipette_id}")

        if position is MountPositions.RIGHT_MOUNT:
            self._set_right_pipette(new_pipette)
            payload = Commands.load_pipette(self._right_pipette)
            response = Commands.send_command(
                command_url=self._get_command_url(), command_dict=payload
            )
            pipette_id = response["data"]["result"]["pipetteId"]
            new_pipette.id = pipette_id
            Log.d(f"Right pipette loaded with ID: {pipette_id}")

    def load_labware(
        self,
        location: DeckLocations,
        labware_definition: Union[str, StandardLabware],
    ) -> None:
        """
        Loads labware into a specified location on the Opentrons Flex robot.

        This method checks if a labware is already present at the specified location. If not, it
        loads the labware based on the provided definition and updates the robot's configuration.
        The unique ID of the loaded labware is then retrieved and assigned.

        Args:
            location (FlexDeckLocations): The location on the deck where the labware will be loaded.
            labware_definition (Union[str, StandardLabware]): The labware definition, either as a
                string (name) or as a `StandardLabware` object, that describes the labware to be loaded.

        Raises:
            Exception: If labware is already loaded at the specified location, an exception is raised.

        Returns:
            None
        """
        Log.i(
            f"Loading labware at location: {location.value} from definition: {labware_definition}"
        )
        labware = Labware(location=location,
                          labware_definition=labware_definition)
        if self.available_labware.get(location) is not None:
            Log.e(f"Labware already loaded at location: {location.value}")
            raise Exception(
                f"Labware {labware.display_name} not available in slot {labware.location.value}."
            )
        self.available_labware[location] = labware
        payload = Commands.load_labware(
            location=location,
            load_name=labware.load_name,
            name_space=labware.name_space,
            version=labware.version,
        )
        response = Commands.send_command(
            command_url=self._get_command_url(), command_dict=payload
        )
        labware.id = response["data"]["result"]["labwareId"]
        Log.d(f"Labware loaded with ID: {labware.id}")

    def pickup_tip(self, labware: Labware, pipette: Pipette) -> str:
        """
        Picks up a tip from a specified labware using the given pipette.

        This method validates that the labware is a tip rack before proceeding. It then sends a
        command to the robot to pick up a tip from the labware with the provided pipette. If the
        labware is not a tip rack, an exception is raised.

        Args:
            labware (FlexLabware): The labware object representing the tip rack from which a tip
                will be picked up.
            pipette (FlexPipette): The pipette object that will pick up the tip from the labware.

        Raises:
            Exception: If the labware is not a tip rack, an exception is raised.

        Returns:
            str: The response from the robot after attempting to pick up the tip, typically a
                success message or status.
        """
        Log.i(
            f"Picking up tip from labware: {labware.display_name} with pipette: {pipette.id}"
        )
        self.validate_configuration(labware=labware, pipette=pipette)
        if not labware.is_tiprack:
            Log.e(
                f"Attempt to pick up tip from non-tiprack labware: {labware.display_name}"
            )
            raise Exception(
                f"Cannot pickup tip from non-tiprack labware {labware.display_name}"
            )
        payload = Commands.pickup_tip(labware=labware, pipette=pipette)
        response = Commands.send_command(
            command_url=self._get_command_url(), command_dict=payload
        )
        Log.d(f"Tip pickup successful. Response: {response}")
        return response

    def aspirate(
        self,
        labware: Labware,
        pipette: Pipette,
        flow_rate: float,
        volume: float,
    ) -> str:
        """
        Aspirates a specified volume of liquid from labware using the provided pipette and flow rate.

        This method logs the aspirate operation details, validates the configuration, and then sends
        a command to the robot to aspirate the specified volume from the labware using the pipette.
        The robot's response is returned after the operation is performed.

        Args:
            labware (FlexLabware): The labware object from which the liquid will be aspirated.
            pipette (FlexPipette): The pipette object that will perform the aspiration.
            flow_rate (float): The flow rate (µL/s) at which the liquid will be aspirated.
            volume (float): The volume (µL) of liquid to aspirate.

        Returns:
            str: The response from the robot after performing the aspiration, typically indicating
                success or providing additional status information.

        Raises:
            Exception: If the configuration of the labware or pipette is invalid (via `validate_configuration`).
        """
        Log.i(
            f"Aspirating {volume} µL at flow rate: {flow_rate} from labware: {labware.display_name} using pipette: {pipette.id}"
        )
        self.validate_configuration(labware=labware, pipette=pipette)

        payload = Commands.aspirate(
            labware=labware, pipette=pipette, flow_rate=flow_rate, volume=volume
        )
        response = Commands.send_command(
            command_url=self._get_command_url(), command_dict=payload
        )
        Log.d(f"Aspirate successful. Response: {response}")
        return response

    def dispense(
        self,
        labware: Labware,
        pipette: Pipette,
        flow_rate: float,
        volume: float,
    ) -> str:
        Log.i(
            f"Aspirating {volume} µL at flow rate: {flow_rate} from labware: {labware.display_name} using pipette: {pipette.id}"
        )
        self.validate_configuration(labware=labware, pipette=pipette)
        payload = Commands.dispense(
            labware=labware, pipette=pipette, flow_rate=flow_rate, volume=volume
        )
        response = Commands.send_command(
            command_url=self._get_command_url(), command_dict=payload
        )
        Log.d(f"Dispense successful. Response: {response}")
        return response

    def blowout(self, labware: Labware, pipette: Pipette, flow_rate: float) -> str:
        """
        Dispenses a specified volume of liquid into labware using the provided pipette and flow rate.

        This method logs the dispense operation details, validates the configuration, and then sends
        a command to the robot to dispense the specified volume into the labware using the pipette.
        The robot's response is returned after the operation is performed.

        Args:
            labware (FlexLabware): The labware object into which the liquid will be dispensed.
            pipette (FlexPipette): The pipette object that will perform the dispensing.
            flow_rate (float): The flow rate (µL/s) at which the liquid will be dispensed.
            volume (float): The volume (µL) of liquid to dispense.

        Returns:
            str: The response from the robot after performing the dispense, typically indicating
                success or providing additional status information.

        Raises:
            Exception: If the configuration of the labware or pipette is invalid (via `validate_configuration`).
        """
        Log.i(
            f"Blowing out tips at flow rate: {flow_rate} from labware: {labware.display_name} using pipette: {pipette.id}"
        )
        self.validate_configuration(labware=labware, pipette=pipette)
        payload = Commands.blowout(
            labware=labware, pipette=pipette, flow_rate=flow_rate
        )
        response = Commands.send_command(
            command_url=self._get_command_url(), command_dict=payload
        )
        Log.d(f"Blowout successful. Response: {response}")
        return response

    def drop_tip(self, labware: Labware, pipette: Pipette):
        """
        Drops the tip from the specified pipette at the given labware location.

        This method logs the tip drop operation, validates the configuration of the
        pipette and labware, and sends a command to the robot to drop the pipette tip
        at the specified labware location. The robot's response is returned after the
        operation is performed.

        Args:
            labware (FlexLabware): The labware object from which the pipette tip will be dropped.
            pipette (FlexPipette): The pipette object that will drop the tip.

        Returns:
            str: The response from the robot after performing the tip drop, typically indicating
                success or providing additional status information.

        Raises:
            Exception: If the configuration of the labware or pipette is invalid (via `validate_configuration`).
        """
        Log.i(
            f"Dropting tip from pipette {pipette.id} at location {labware.display_name}"
        )
        self.validate_configuration(labware=labware, pipette=pipette)
        payload = Commands.drop_tip(labware=labware, pipette=pipette)
        response = Commands.send_command(
            command_url=self._get_command_url(), command_dict=payload
        )
        Log.d(f"Drop successful. Response: {response}")
        return response

    def move_to_coordiantes(
        self,
        pipette: Pipette,
        x: float,
        y: float,
        z: float,
        min_z_height: float,
        force_direct: bool,
    ):
        """
        Moves the specified pipette to the given (X, Y, Z) coordinates with optional Z height limitation.

        This method logs the movement request, validates the configuration of the pipette,
        and sends a command to the robot to move the pipette to the specified coordinates.
        The movement is constrained by a minimum Z height and can be forced to proceed
        directly if specified. The response from the robot after the movement is performed is returned.

        Args:
            pipette (FlexPipette): The pipette object that will be moved.
            x (float): The target X-coordinate for the pipette's movement.
            y (float): The target Y-coordinate for the pipette's movement.
            z (float): The target Z-coordinate for the pipette's movement.
            min_z_height (float): The minimum Z height constraint for the movement.
            force_direct (bool): If True, the movement will be forced directly to the coordinates,
                                bypassing any additional checks or adjustments.

        Returns:
            str: The response from the robot after the movement, typically indicating
                success or additional status information.

        Raises:
            Exception: If the configuration of the pipette is invalid (via `validate_configuration`).
        """
        Log.i(
            f"Moving pipette {pipette.id} to coordinates (X, Y, Z, Z-Lmit): {x}, {y}, {z}, {min_z_height}, force-direct={force_direct}"
        )
        self.validate_configuration(labware=None, pipette=pipette)
        payload = Commands.move_to_coordinates(
            pipette=pipette,
            x=x,
            y=y,
            z=z,
            min_z_height=min_z_height,
            force_direct=force_direct,
        )
        response = Commands.send_command(
            command_url=self._get_command_url(), command_dict=payload
        )
        Log.d(f"Move to coordinates successful. Response: {response}")
        return response

    def move_to_well(self, labware: Labware, pipette: Pipette):
        """
        Moves the specified pipette to a specific well in the given labware.

        This method logs the movement request, validates the configuration of the labware
        and pipette, and sends a command to the robot to move the pipette to the specified
        well within the labware. The response from the robot after the movement is performed is returned.

        Args:
            labware (FlexLabware): The labware object containing the well to which the pipette will be moved.
            pipette (FlexPipette): The pipette object that will be moved to the well.

        Returns:
            str: The response from the robot after the movement, typically indicating success or additional status information.

        Raises:
            Exception: If the configuration of the labware or pipette is invalid (via `validate_configuration`).
        """
        Log.i(
            f"Moving pipette {pipette.id} to labware {labware.display_name} at well location {labware.location.value}"
        )
        self.validate_configuration(labware=labware, pipette=pipette)
        payload = Commands.move_to_well(labware=labware, pipette=pipette)
        response = Commands.send_command(
            command_url=self._get_command_url(), command_dict=payload
        )
        Log.d(f"Move to well successful. Response: {response}")
        return response

    def move_relative(self, pipette: Pipette, distance: float, axis: Axis):
        """
        Moves the specified pipette a relative distance along a specified axis.

        This method logs the request for the relative movement, validates the pipette's configuration,
        and sends a command to the robot to move the pipette along the specified axis by the given distance.
        The response from the robot after the movement is performed is returned.

        Args:
            pipette (FlexPipette): The pipette object that will be moved.
            distance (float): The distance in millimeters the pipette will move along the specified axis.
            axis (FlexAxis): The axis along which the pipette will move (e.g., X, Y, or Z).

        Returns:
            str: The response from the robot after the relative move, typically indicating success or additional status information.

        Raises:
            Exception: If the configuration of the pipette is invalid (via `validate_configuration`).
        """
        Log.i(
            f"Relative move of pipette {pipette.id} {distance}mm along {axis.value} axis"
        )
        self.validate_configuration(pipette=pipette)
        payload = Commands.move_relative(
            pipette=pipette, distance=distance, axis=axis)
        response = Commands.send_command(
            command_url=self._get_command_url(), command_dict=payload
        )
        Log.d(f"Relative move successful. Response: {response}")
        return response

    def run_protocol(self, protocol_name: str) -> str:
        """
        Runs a protocol by its name.

        This method checks if the specified protocol is available, retrieves its ID,
        and initiates the protocol run. If successful, the method logs the run's details
        and returns the response from the run. If an error occurs, it logs the error
        and raises an exception.

        Args:
            protocol_name (str): The name of the protocol to run.

        Returns:
            str: The response from the run, typically containing details of the protocol execution.

        Raises:
            ValueError: If the protocol is not available.
            Exception: If the protocol run fails during execution.
        """
        protocol = self.available_protocols.get(protocol_name)
        if protocol is None:
            Log.e(f"Protocol '{protocol_name}' not available.")
            raise ValueError(f"Protocol '{protocol_name}' not available.")
        protocol_id = protocol.get("id")
        Log.i(
            f"Setting up '{protocol_name}' protocol for run with ID: {protocol_id}")
        try:
            run_id = Runs.run_protocol(
                runs_url=self._get_runs_url(), protocol_id=protocol_id
            )
            response = self.play_run(run_id)
            Log.i(f"Protocol {protocol_id} running under {run_id}. ")
            return response["data"]["id"]
        except Exception as e:
            Log.e(f"Failed to run protocol with ID {protocol_id}: {e}")
            raise

    def delete_protocol(self, protocol_name: str) -> str:
        """
        Deletes a protocol by its name.

        This method looks for the protocol in the available protocols list, and if found,
        it deletes the protocol using its ID by calling the `FlexRuns.delete_protocol` method.
        If the protocol is not found, it raises a `ValueError`. It also logs the success or failure
        of the deletion process.

        Args:
            protocol_name (str): The name of the protocol to be deleted.

        Returns:
            str: The response from the deletion operation, typically indicating success or failure.

        Raises:
            ValueError: If the protocol with the specified name is not available.
            Exception: If the deletion operation fails.
        """
        protocol = self.available_protocols.get(protocol_name)
        if protocol is None:
            Log.e(f"Protocol '{protocol_name}' not available.")
            raise ValueError(f"Protocol '{protocol_name}' not available.")
        protocol_id = protocol.get("id")
        Log.i(f"Deleting '{protocol_name}' protocol with ID: {protocol_id}")
        try:
            response = Runs.delete_protocol(
                protocols_url=self._get_protocols_url(), protocol_id=protocol_id
            )
            self.update_available_protocols()
            Log.i(f"Protocol {protocol_id} deleted successfully. ")
            return response
        except Exception as e:
            Log.e(f"Failed to delete protocol with ID {protocol_id}: {e}")
            raise

    def upload_protocol(self, protocol_file_path: str) -> str:
        """
        Deletes a protocol by its name.

        This method checks if the specified protocol is available, retrieves its ID,
        and deletes it. If the deletion is successful, the available protocols are updated
        and the method returns the response from the deletion. If an error occurs,
        it logs the error and raises an exception.

        Args:
            protocol_name (str): The name of the protocol to delete.

        Returns:
            str: The response from the deletion, typically containing details of the deletion process.

        Raises:
            ValueError: If the protocol is not available.
            Exception: If the protocol deletion fails during execution.
        """
        Log.i(f"Uploading protocol from file: {protocol_file_path}")
        if not os.path.exists(protocol_file_path):
            Log.e(f"Protocol file path does not exist: {protocol_file_path}")
            raise Exception(
                f"Protocol path {protocol_file_path} does not exist")

        try:
            response = Runs.upload_protocol(
                protocols_url=self._get_protocols_url(),
                protocol_file_path=protocol_file_path,
            )
            self.update_available_protocols()
            Log.i("Protocol uploaded successfully.")
            return response
        except Exception as e:
            Log.e(
                f"Failed to upload protocol from file {protocol_file_path}: {e}")
            raise

    def upload_protocol_custom_labware(
        self, protocol_file_path: str, *custom_labware_file_paths: str
    ) -> str:
        """
        Uploads a protocol file with custom labware files.

        This method checks if the provided protocol file and all custom labware files exist.
        If they exist, the protocol and labware files are uploaded. After the upload,
        the available protocols are updated. If any file does not exist or if the upload fails,
        an error is logged and an exception is raised.

        Args:
            protocol_file_path (str): The file path of the protocol to upload.
            custom_labware_file_paths (str): One or more file paths of custom labware to upload.

        Returns:
            str: The response from the upload, typically containing the status of the upload process.

        Raises:
            Exception: If any of the file paths do not exist or if the upload fails.
        """
        Log.i(
            f"Uploading protocol from file: {protocol_file_path} with custom labware from files: {', '.join(custom_labware_file_paths)}"
        )

        # Check if protocol file and all custom labware files exist
        if not os.path.exists(protocol_file_path):
            Log.e(f"Protocol file path does not exist: {protocol_file_path}")
            raise Exception(
                f"Protocol file path does not exist: {protocol_file_path}")

        for labware_file_path in custom_labware_file_paths:
            if not os.path.exists(labware_file_path):
                Log.e(
                    f"Custom labware file path does not exist: {labware_file_path}")
                raise Exception(
                    f"Custom labware file path does not exist: {labware_file_path}"
                )

        try:
            response = Runs.upload_protocol_custom_labware(
                protocols_url=self._get_protocols_url(),
                protocol_file_path=protocol_file_path,
                labware_file_paths=list(
                    custom_labware_file_paths),  # Updated parameter
            )
            self.update_available_protocols()
            Log.i("Protocol uploaded with custom labware successfully.")
            return response
        except Exception as e:
            Log.e(
                f"Failed to upload protocol from file {protocol_file_path} with custom labware from files {', '.join(custom_labware_file_paths)}: {e}"
            )
            raise

    def get_protocol_list(self) -> str:
        """
        Retrieves a list of available protocols.

        This method fetches the list of protocols from the specified URL. If the fetch is
        successful, the list is returned. If an error occurs during the process, an error is
        logged and an exception is raised.

        Returns:
            str: The response containing the protocol list, typically in JSON format.

        Raises:
            Exception: If the attempt to fetch the protocol list fails.
        """
        Log.i("Fetching protocol list.")
        try:
            response = Runs.get_protocols_list(
                protocols_url=self._get_protocols_url())
            Log.i(f"Retrieved protocol list successfully")
            return response
        except Exception as e:
            Log.e(f"Failed to fetch protocol list: {e}")
            raise

    def update_available_protocols(self) -> None:
        """
        Updates the list of available protocols and stores the most recent version of each protocol.

        This method fetches the list of protocols, processes each protocol entry, and updates
        the internal `available_protocols` dictionary. If multiple versions of a protocol exist,
        it stores the most recent version based on the `createdAt` timestamp.

        The `available_protocols` dictionary will be populated with the protocol names as keys,
        and their corresponding protocol IDs and creation timestamps as values.

        Returns:
            dict: A dictionary mapping protocol names to their corresponding protocol IDs.

        Raises:
            Exception: If fetching the protocol list fails or the protocol entries are malformed.
        """
        Log.i("Updating available protocols.")
        self.available_protocols = {}
        all_protocols = self.get_protocol_list()
        for entry in all_protocols:
            protocol_name = entry["metadata"]["protocolName"]
            protocol_id = entry["id"]
            created_at = datetime.fromisoformat(
                entry["createdAt"].replace("Z", "+00:00")
            )

            # Check if protocol already exists in the dictionary
            if protocol_name not in self.available_protocols:
                self.available_protocols[protocol_name] = {
                    "id": protocol_id,
                    "createdAt": created_at,
                }
            else:
                # Compare the dates and store the most recent one
                if created_at > self.available_protocols[protocol_name]["createdAt"]:
                    self.available_protocols[protocol_name] = {
                        "id": protocol_id,
                        "createdAt": created_at,
                    }

        # Extract only protocol names and their corresponding IDs
        result = {name: data["id"]
                  for name, data in self.available_protocols.items()}

        return result

    def delete_run(self, run_id: int) -> str:
        """
        Deletes a specific run based on the provided run ID.

        This method attempts to delete a run using the provided `run_id`. It communicates with
        the FlexRuns service to delete the run and logs the result. If the deletion is successful,
        it returns the response from the service.

        Args:
            run_id (int): The ID of the run to delete.

        Returns:
            str: The response from the FlexRuns service after attempting to delete the run.

        Raises:
            Exception: If the deletion fails due to a service error or invalid run ID.
        """
        Log.i(f"Deleting run with ID: {run_id}")
        try:
            response = Runs.delete_run(
                runs_url=self._get_runs_url(), run_id=run_id)
            Log.i(f"Run {run_id} deleted successfully. ")
            return response
        except Exception as e:
            Log.e(f"Failed to delete run with ID {run_id}: {e}")
            raise

    def get_run_status(self, run_id: str) -> str:
        """
        Retrieves the status of a specific run based on the provided run ID.

        This method communicates with the FlexRuns service to fetch the current status
        of a run using the `run_id`. If successful, it logs and returns the response
        containing the status. If an error occurs, it logs the error and raises an exception.

        Args:
            run_id (int): The ID of the run whose status is to be retrieved.

        Returns:
            str: The status of the run, as returned by the FlexRuns service.

        Raises:
            Exception: If the status retrieval fails due to a service error or invalid run ID.
        """
        Log.i(f"Fetching status for run ID: {run_id}")
        try:
            response = Runs.get_run_status(
                runs_url=self._get_runs_url(), run_id=run_id)
            Log.i(f"Status for run {run_id} retrieved successfully. ")
            return response
        except Exception as e:
            Log.e(f"Failed to fetch status for run ID {run_id}: {e}")
            raise

    def get_run_list(self) -> str:
        """
        Retrieves the list of all runs from the FlexRuns service.

        This method communicates with the FlexRuns service to fetch the list of all
        available runs. If successful, it logs and returns the response containing
        the list of runs. If an error occurs, it logs the error and raises an exception.

        Returns:
            str: The list of runs, as returned by the FlexRuns service.

        Raises:
            Exception: If the run list retrieval fails due to a service error.
        """
        Log.i("Fetching run list.")
        try:
            response = Runs.get_runs_list(runs_url=self._get_runs_url())
            Log.i("Run list retrieved successfully. ")
            return response
        except Exception as e:
            Log.e(f"Failed to fetch run list: {e}")
            raise

    def pause_run(self, run_id: int) -> str:
        """
        Pauses a running protocol with the given run ID.

        This method communicates with the FlexRuns service to pause the run with the
        specified `run_id`. If the operation is successful, it logs the success and
        returns the response. If an error occurs, it logs the error and raises an exception.

        Args:
            run_id (int): The ID of the run to pause.

        Returns:
            str: The response from the FlexRuns service indicating the result of the pause operation.

        Raises:
            Exception: If the run pausing operation fails due to a service error.
        """
        Log.i(f"Pausing run with ID: {run_id}")
        try:
            response = Runs.pause_run(
                runs_url=self._get_runs_url(), run_id=run_id)
            Log.i(f"Run {run_id} paused successfully. ")
            return response
        except Exception as e:
            Log.e(f"Failed to pause run with ID {run_id}: {e}")
            raise

    def play_run(self, run_id: int) -> str:
        """
        Starts the execution of a protocol run with the given run ID.

        This method communicates with the FlexRuns service to start the run with the
        specified `run_id`. If the operation is successful, it logs the success and
        returns the response. If an error occurs, it logs the error and raises an exception.

        Args:
            run_id (int): The ID of the run to start.

        Returns:
            str: The response from the FlexRuns service indicating the result of the play operation.

        Raises:
            Exception: If the run playing operation fails due to a service error.
        """
        Log.i(f"Playing run with ID: {run_id}")
        try:
            response = Runs.play_run(
                runs_url=self._get_runs_url(), run_id=run_id)
            Log.i(f"Run {run_id} started successfully. ")
            return response
        except Exception as e:
            Log.e(f"Failed to play run with ID {run_id}: {e}")
            raise

    def stop_run(self, run_id: str) -> str:
        """
        Stops the execution of a protocol run with the given run ID.

        This method communicates with the FlexRuns service to stop the run with the
        specified `run_id`. If the operation is successful, it logs the success and
        returns the response. If an error occurs, it logs the error and raises an exception.

        Args:
            run_id (str): The ID of the run to stop.

        Returns:
            str: The response from the FlexRuns service indicating the result of the stop operation.

        Raises:
            Exception: If the run stopping operation fails due to a service error.
        """
        Log.i(f"Stopping run with ID: {run_id}")
        try:
            response = Runs.stop_run(
                runs_url=self._get_runs_url(), run_id=run_id)
            Log.i(f"Run {run_id} stopped successfully. ")
            return response
        except Exception as e:
            Log.e(f"Failed to stop run with ID {run_id}: {e}")
            raise

    def resume_run(self, run_id: str) -> str:
        """
        Resumes the execution of a protocol run with the given run ID.

        This method communicates with the FlexRuns service to resume the run with the
        specified `run_id`. If the operation is successful, it logs the success and
        returns the response. If an error occurs, it logs the error and raises an exception.

        Args:
            run_id (str): The ID of the run to restume.

        Returns:
            str: The response from the FlexRuns service indicating the result of the stop operation.

        Raises:
            Exception: If the run resume operation fails due to a service error.
        """
        Log.i(f"Resuming run with ID: {run_id}")
        try:
            response = Runs.resume_run(
                runs_url=self._get_runs_url(), run_id=run_id)
            Log.i(f"Run {run_id} stopped successfully. ")
            return response
        except Exception as e:
            Log.e(f"Failed to stop run with ID {run_id}: {e}")
            raise

    def lights_on(self) -> str:
        """
        Turns the lights on.

        This method sends a command to turn the lights on by interacting with the
        FlexLights service. If successful, it logs the success and returns the response.
        If an error occurs, it logs the error and raises an exception.

        Returns:
            str: The response from the FlexLights service indicating the result of the light-on operation.

        Raises:
            Exception: If the light turning-on operation fails due to a service error.
        """
        Log.i("Turning lights on.")
        try:
            self.light_state = Lights.ON
            response = Runs.set_lights(
                lights_url=self._get_lights_url(), light_status=Lights.ON.value
            )
            Log.i("Lights turned on successfully.")
            return response
        except Exception:
            Log.e("Failed to turn lights on.")
            raise

    def lights_off(self) -> str:
        """
        Turns the lights off.

        This method sends a command to turn the lights off by interacting with the
        FlexLights service. If successful, it logs the success and returns the response.
        If an error occurs, it logs the error and raises an exception.

        Returns:
            str: The response from the FlexLights service indicating the result of the light-off operation.

        Raises:
            Exception: If the light turning-off operation fails due to a service error.
        """
        Log.i("Turning lights off.")
        try:
            self.light_state = Lights.OFF
            response = Runs.set_lights(
                lights_url=self._get_lights_url(), light_status=Lights.OFF.value
            )
            Log.i("Lights turned off successfully.")
            return response
        except Exception:
            Log.e("Failed to turn lights off.")
            raise

    def flash_lights(self, number_of_times: int) -> str:
        """
        Flashes the lights a specified number of times.

        This method flashes the lights by turning them on and off with a 0.5-second
        delay between each toggle. The lights will flash the number of times
        specified by the `number_of_times` parameter. If an error occurs, it logs the
        error and raises an exception.

        Args:
            number_of_times (int): The number of times to flash the lights.

        Returns:
            str: A string indicating the completion of the flash sequence. This method does not return a response directly,
                but logs the actions taken.

        Raises:
            ValueError: If the number of times requested to flash the lights is less than 1.
            Exception: If the flashing sequence fails due to an error in turning the lights on or off.
        """
        if number_of_times < 1:
            Log.e(f"Cannot flash lights {number_of_times}.")
            raise ValueError(f"Cannot flash lights {number_of_times}.")
        Log.i(f"Flashing lights {number_of_times} times.")
        try:
            for _ in range(number_of_times):
                self.lights_on()
                time.sleep(0.5)
                self.lights_off()
                time.sleep(0.5)
        except Exception as e:
            Log.e("Failed flashing lights")
            raise

    def lights_status(self) -> str:
        """
        Fetches the current status of the lights.

        This method retrieves the current status of the lights from the system.
        If the request fails, it logs the error and raises an exception.

        Returns:
            str: A string representing the current status of the lights.

        Raises:
            Exception: If the request to fetch the lights status fails.
        """
        Log.i("Fetching lights status.")
        try:
            response = Runs.get_lights(self._get_lights_url())
            Log.i("Lights status retrieved successfully. ")
            return response
        except Exception as e:
            Log.e("Failed to fetch lights status.")
            raise

    def create_run(self) -> str:
        """
        Creates a new run from the protocol.

        This method creates a new run based on the specified protocol, logs the process,
        and sets the run ID upon successful creation.

        Returns:
            str: The ID of the newly created run.

        Raises:
            Exception: If the run creation fails.
        """
        Log.i("Creating a new run.")
        try:
            response = Runs.create_run_from_protocol(self._get_runs_url())
            run_id = response["data"]["id"]
            self._set_run_id(run_id)
            Log.i(f"New run created successfully with ID: {run_id}")
            return run_id
        except Exception:
            Log.e("Failed to create a new run.")
            raise

    def home_robot(self) -> str:
        """
        Sends a command to home the entire robot.

        This method sends a "home" command to the system. If the command is
        executed successfully, it logs the success message. If there is an error,
        it logs the failure and raises an exception.

        Returns:
            str: The response from the system after executing the home command.

        Raises:
            Exception: If the command to home the system fails.
        """
        Log.i("Sending home command.")
        try:
            payload = Commands.home_robot()
            response = Commands.send_command(
                command_url=self._get_home_url(), command_dict=payload
            )
            return response
        except Exception:
            Log.e("Failed to execute home command.")
            raise

    def validate_configuration(
        self, labware: Labware = None, pipette: Pipette = None
    ) -> None:
        """
        Validates the configuration of labware and pipette.

        This method checks if the provided labware is available in the configured
        slot and if the specified pipette is correctly mounted. If either check fails,
        an exception is raised with an appropriate error message.

        Args:
            labware (FlexLabware, optional): The labware to validate. Defaults to None.
            pipette (FlexPipette, optional): The pipette to validate. Defaults to None.

        Raises:
            Exception: If the labware is not available in the specified slot or if
            the pipette is not mounted correctly.
        """
        Log.i("Validating configuration for labware and pipette.")

        try:
            if labware is None or self.available_labware.get(labware.location) is None:
                error_message = (
                    f"Labware {labware.display_name} not available in slot "
                    f"{labware.location.value}."
                )
                Log.e(error_message)
                raise Exception(error_message)

            if (
                pipette is None
                or self.gantry.get(pipette.mount_position).id != pipette.id
            ):
                error_message = f"Pipette {pipette.pipette} not mounted."
                Log.e(error_message)
                raise Exception(error_message)

            Log.i("Configuration validated successfully.")

        except Exception as e:
            Log.e(f"Validation failed: {e}.")
            raise

    def find_ip(self) -> str:
        """
        Finds the IPv4 address of the robot by matching its MAC address in the ARP table.

        This method retrieves the ARP table using the `arp -a` command, parses the table,
        and matches the MAC address of the robot to extract the corresponding IPv4 address.

        Returns:
            str: The IPv4 address of the robot if found.

        Raises:
            Exception: If there is an error retrieving the ARP table or the IPv4 address
                    cannot be determined or if the IPv4 is not found in the ARP table.
        """
        result = subprocess.run(["arp", "-a"], capture_output=True, text=True)
        if result.returncode != 0:
            Log.e("Error retrieving ARP table")
            raise Exception("Error retrieving ARP table")

        robot_mac = self._get_robot_mac_address().lower()
        for line in result.stdout.splitlines():
            if "dynamic" in line:
                parts = line.split()
                if parts[1].lower() == robot_mac:
                    return parts[0]

        Log.e("MAC address not found in ARP table.")
        raise Exception("MAC address not found in ARP table.")

    def _set_runs_url(self, runs_url: str) -> None:
        self._runs_url = runs_url

    def _set_base_url(self, base_url: str) -> None:
        self._base_url = base_url

    def _set_home_url(self, home_url: str) -> None:
        self._home_url = home_url

    def _set_command_url(self, command_url: str) -> None:
        self._command_url = command_url

    def _set_protocols_url(self, protocols_url: str) -> None:
        self._protocols_url = protocols_url

    def _set_lights_url(self, lights_url: str) -> None:
        self._lights_url = lights_url

    def _set_run_id(self, run_id: str) -> None:
        self._run_id = run_id

    def _set_robot_ipv4(self, ipv4: str) -> None:
        # ipv4_regex = r"^((25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])\.){3}(25[0-5]|2[0-4][0-9]|[1-9]?[0-9])$"
        # if not re.match(ipv4_regex, ipv4):
        #     Log.e(f"Invalid IPv4 address: {ipv4}" )
        #     raise ValueError(f"Invalid IPv4 address: {ipv4}")

        # Attempt to ping the IP address to check if it is reachable
        try:
            result = subprocess.run(
                # Use `-n 1` for Windows compatibility
                ["ping", "-n", "1", ipv4],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if result.returncode != 0:
                Log.e(f"Cannot communicate with IP address: {ipv4}")
                raise ConnectionError(
                    f"Cannot communicate with IP address: {ipv4}")
        except Exception as e:
            Log.e(
                f"Error during communication check for IP address {ipv4}: {e}")
            raise

        self._robot_ipv4 = ipv4

    def _set_robot_mac_address(self, mac_address: str) -> None:
        mac_regex = (
            "^([0-9A-Fa-f]{2}[:-])"
            + "{5}([0-9A-Fa-f]{2})|"
            + "([0-9a-fA-F]{4}\\."
            + "[0-9a-fA-F]{4}\\."
            + "[0-9a-fA-F]{4})$"
        )
        if not re.match(mac_regex, mac_address):
            Log.e(f"Invalid MAC address: {mac_address}")
            raise ValueError(f"Invalid MAC address: {mac_address}")

        self._robot_mac_address = mac_address.replace(":", "-")

    def _set_left_pipette(self, pipette: Pipette) -> None:
        current_pipette = self.gantry.get(MountPositions.LEFT_MOUNT)
        if current_pipette is None:
            self.gantry[MountPositions.LEFT_MOUNT] = pipette
            self._left_pipette = pipette
        else:
            raise Exception(
                f"Gantry mount position {MountPositions.LEFT_MOUNT.value} is occupied by {current_pipette.get_pipette()}."
            )

    def _set_right_pipette(self, pipette: Pipette) -> None:
        current_pipette = self.gantry.get(MountPositions.RIGHT_MOUNT)
        if current_pipette is None:
            self.gantry[MountPositions.RIGHT_MOUNT] = pipette
            self._right_pipette = pipette
        else:
            raise Exception(
                f"Gantry mount position {MountPositions.RIGHT_MOUNT.value} is occupied by {current_pipette.get_pipette()}."
            )

    def _get_robot_ipv4(self) -> str:
        return self._robot_ipv4

    def _get_robot_mac_address(self) -> str:
        return self._robot_mac_address

    def _get_runs_url(self) -> str:
        return self._runs_url

    def _get_home_url(self) -> str:
        return self._home_url

    def _get_base_url(self) -> str:
        return self._base_url

    def _get_command_url(self) -> str:
        return self._command_url

    def _get_protocols_url(self) -> str:
        return self._protocols_url

    def _get_lights_url(self) -> str:
        return self._lights_url

    def _get_run_id(self) -> str:
        return self._run_id

    def _get_left_pipette(self) -> Pipette:
        return self.gantry.get(MountPositions.LEFT_MOUNT)

    def _get_right_pipette(self) -> Pipette:
        return self.gantry.get(MountPositions.RIGHT_MOUNT)

    def _get_available_labware(self) -> dict:
        return self.available_labware
