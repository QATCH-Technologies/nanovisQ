"""
commands.py

This module provides the `Commands` class, which contains factory methods
for constructing JSON-compatible command payloads for the Opentrons Flex system.
Each static method generates a specific command (e.g., load labware, pick up tips,
aspirate, dispense), and `_create_base_command` centralizes common payload structure.

The `send_command` utility will transmit the payload to a robot server endpoint and
handle HTTP responses.


Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2025-08-07

Version:
    1.0
"""

import json
import requests

try:
    from src.constants import (
        CommandType,
        Intents,
        Axis,
        DeckLocations,
        SlotName,
        HEADERS,
    )
    from src.pipette import Pipette
    from src.labware import Labware

    class Log:
        """
        Simple logger implementation for standalone FluxControls usage.

        Methods:
            d: Debug-level log.
            i: Info-level log.
            w: Warning-level log.
            e: Error-level log.
        """

        def d(tag, msg=""):
            print("DEBUG:", tag, msg)

        def i(tag, msg=""):
            print("INFO:", tag, msg)

        def w(tag, msg=""):
            print("WARNING:", tag, msg)

        def e(tag, msg=""):
            print("ERROR:", tag, msg)

    Log.i("Running FluxControls as standalone app")

except (ImportError, ModuleNotFoundError):
    from QATCH.FluxControls.src.constants import (
        CommandType,
        Intents,
        Axis,
        DeckLocations,
        SlotName,
        HEADERS,
    )
    from QATCH.FluxControls.src.pipette import Pipette
    from QATCH.FluxControls.src.labware import Labware
    from QATCH.common.logger import Logger as Log


class Commands:
    """
    Collection of static methods for building and sending FluxControls robot commands.
    """

    @staticmethod
    def _create_base_command(
        command_type: CommandType, params: dict, intents: Intents = None
    ) -> dict:
        """
        Build the base command dictionary with common data fields.

        Args:
            command_type (CommandType): Enum member indicating the command action.
            params (dict): Specific parameters for the command.
            intents (Intents): Enum member indicating the intent category.

        Returns:
            dict: A structured command payload ready for JSON serialization.
        """
        if intents:
            return {
                "data": {
                    "commandType": command_type.value,
                    "params": params,
                    "intent": intents.value,
                }
            }
        else:
            return {
                "data": {
                    "commandType": command_type.value,
                    "params": params,
                }
            }

    @staticmethod
    def load_labware(
        location: DeckLocations, load_name: str, name_space: str, version: int
    ) -> dict:
        """
        Construct a command to load labware into a specified deck slot.

        Args:
            location (DeckLocations): Deck slot enum where the labware will be placed.
            load_name (str): Identifier/name of the labware definition to load.
            name_space (str): Namespace or module where the labware definition lives.
            version (int): Version number of the labware definition.

        Returns:
            dict: Payload for the LOAD_LABWARE command.
        """
        params = {
            "location": SlotName.get_slot_name(location),
            "loadName": load_name,
            "namespace": name_space,
            "version": version,
        }
        return Commands._create_base_command(
            CommandType.LOAD_LABWARE, params, Intents.SETUP
        )

    @staticmethod
    def load_pipette(pipette: Pipette) -> dict:
        """
        Construct a command to load a pipette onto the robot.

        Args:
            pipette (Pipette): Pipette instance containing mount and identity information.

        Returns:
            dict: Payload for the LOAD_PIPETTE command.
        """
        params = {
            "pipetteName": pipette.pipette,
            "mount": pipette.mount_position,
        }
        return Commands._create_base_command(
            CommandType.LOAD_PIPETTE, params, Intents.SETUP
        )

    @staticmethod
    def pickup_tip(labware: Labware, pipette: Pipette) -> dict:
        """
        Construct a command for the pipette to pick up a tip from labware.

        Args:
            labware (Labware): Labware instance representing the tip rack.
            pipette (Pipette): Pipette instance to perform the tip pickup.

        Returns:
            dict: Payload for the PICKUP_TIP command.
        """
        params = {
            "labwareId": labware.id,
            "wellName": labware.location,
            "wellLocation": {"origin": "top", "offset": labware.get_offsets()},
            "pipetteId": pipette.id,
        }
        return Commands._create_base_command(
            CommandType.PICKUP_TIP, params, Intents.SETUP
        )

    @staticmethod
    def aspirate(
        labware: Labware, pipette: Pipette, flow_rate: float, volume: float
    ) -> dict:
        """
        Construct a command for the pipette to aspirate liquid from a well.

        Args:
            labware (Labware): Labware instance containing the target well.
            pipette (Pipette): Pipette instance performing the aspiration.
            flow_rate (float): Speed of aspiration in microliters per second.
            volume (float): Volume of liquid to aspirate in microliters.

        Returns:
            dict: Payload for the ASPIRATE command.
        """
        params = {
            "labwareId": labware.id,
            "wellName": labware.location,
            "wellLocation": {"origin": "top", "offset": labware.get_offsets()},
            "flowRate": flow_rate,
            "volume": volume,
            "pipetteId": pipette.id,
        }
        return Commands._create_base_command(
            CommandType.ASPIRATE, params, Intents.SETUP
        )

    @staticmethod
    def dispense(
        labware: Labware, pipette: Pipette, flow_rate: float, volume: float
    ) -> dict:
        """
        Construct a command for the pipette to dispense liquid into a well.

        Args:
            labware (Labware): Labware instance containing the target well.
            pipette (Pipette): Pipette instance performing the dispense.
            flow_rate (float): Speed of dispensing in microliters per second.
            volume (float): Volume of liquid to dispense in microliters.

        Returns:
            dict: Payload for the DISPENSE command.
        """
        params = {
            "labwareId": labware.id,
            "wellName": labware.location,
            "wellLocation": {"origin": "top", "offset": labware.get_offsets()},
            "flowRate": flow_rate,
            "volume": volume,
            "pipetteId": pipette.id,
        }
        return Commands._create_base_command(
            CommandType.DISPENSE, params, Intents.SETUP
        )

    @staticmethod
    def blowout(labware: Labware, pipette: Pipette, flow_rate: float) -> dict:
        """
        Construct a command to blow out residual liquid from the pipette tip.

        Args:
            labware (Labware): Labware instance containing the target well for blowout.
            pipette (Pipette): Pipette instance performing the blowout.
            flow_rate (float): Flow rate for blowout in microliters per second.

        Returns:
            dict: Payload for the BLOWOUT command.
        """
        params = {
            "labwareId": labware.id,
            "wellName": labware.location,
            "wellLocation": {"origin": "top", "offset": labware.get_offsets()},
            "flowRate": flow_rate,
            "pipetteId": pipette.id,
        }
        return Commands._create_base_command(CommandType.BLOWOUT, params, Intents.SETUP)

    @staticmethod
    def drop_tip(labware: Labware, pipette: Pipette) -> dict:
        """
        Construct a command for the pipette to drop the currently held tip.

        Args:
            labware (Labware): Labware instance representing the drop location (waste or rack).
            pipette (Pipette): Pipette instance dropping the tip.

        Returns:
            dict: Payload for the DROP_TIP command.
        """
        params = {
            "labwareId": labware.id,
            "wellName": labware.location,
            "wellLocation": {"origin": "top", "offset": labware.get_offsets()},
            "pipetteId": pipette.id,
        }
        return Commands._create_base_command(
            CommandType.DROP_TIP, params, Intents.SETUP
        )

    @staticmethod
    def move_to_well(labware: Labware, pipette: Pipette) -> dict:
        """
        Construct a command to move the pipette to the top of a specified well.

        Args:
            labware (Labware): Labware instance containing the target well.
            pipette (Pipette): Pipette instance to move.

        Returns:
            dict: Payload for the MOVE_TO_WELL command.
        """
        params = {
            "labwareId": labware.id,
            "wellName": labware.location,
            "wellLocation": {"origin": "top", "offset": labware.get_offsets()},
            "pipetteId": pipette.id,
        }
        return Commands._create_base_command(
            CommandType.MOVE_TO_WELL, params, Intents.SETUP
        )

    @staticmethod
    def move_to_coordinates(
        pipette: Pipette,
        x: float,
        y: float,
        z: float,
        min_z_height: float,
        force_direct: bool,
    ) -> dict:
        """
        Construct a command to move the pipette to absolute X/Y/Z coordinates.

        Args:
            pipette (Pipette): Pipette instance to move.
            x (float): Target X-coordinate in millimeters.
            y (float): Target Y-coordinate in millimeters.
            z (float): Target Z-coordinate in millimeters.
            min_z_height (float): Minimum safe Z height during XY travel.
            force_direct (bool): Whether to move directly without toolpath planning.

        Returns:
            dict: Payload for the MOVE_TO_WELL command (absolute move).
        """
        params = {
            "coordinates": {"x": x, "y": y, "z": z},
            "minimumZHeight": min_z_height,
            "forceDirect": force_direct,
            "pipetteId": pipette.id,
        }
        return Commands._create_base_command(
            CommandType.MOVE_TO_WELL, params, Intents.SETUP
        )

    @staticmethod
    def move_relative(pipette: Pipette, distance: float, axis: Axis) -> dict:
        """
        Construct a command to move the pipette by a relative distance along an axis.

        Args:
            pipette (Pipette): Pipette instance to move.
            distance (float): Distance to move in millimeters.
            axis (Axis): Enum indicating which axis (X, Y, or Z) to move along.

        Returns:
            dict: Payload for the MOVE_TO_WELL command (relative move).
        """
        params = {"axis": axis.value,
                  "distance": distance, "pipetteId": pipette.id}
        return Commands._create_base_command(
            CommandType.MOVE_TO_WELL, params, Intents.SETUP
        )

    @staticmethod
    def home_robot() -> dict:
        """
        Creates the command to home the entire robot

        Returns:    
            dict: Payload for the HOME command.
        """
        params = {}
        return Commands._create_base_command(
            command_type=CommandType.HOME, params=params)

    @staticmethod
    def send_command(command_url: str, command_dict: dict) -> dict:
        """
        Send a constructed command payload to the robot server endpoint.

        Args:
            command_url (str): Full URL of the robot command API endpoint.
            command_dict (dict): Payload generated by other Commands methods.

        Returns:
            dict: Parsed JSON response from the server if successful.
            None: If an HTTP error or network exception occurs.
        """
        payload = json.dumps(command_dict)
        headers = {"Content-Type": "application/json"}
        headers.update(HEADERS)

        try:
            response = requests.post(
                command_url, headers=headers, data=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            Log.e("send_command", f"Error sending request: {e}")
            return None
