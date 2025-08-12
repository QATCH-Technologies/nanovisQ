"""
constants.py

This module contains constants and enumerations used for managing and interacting with the Opentrons Flex device.
It defines configuration details, hardware components, and operational commands.

Constants:
----------
    HTTP_PORT (str): Default HTTP port for communication.
    HEADERS (dict): HTTP headers specifying the Opentrons version.
    CONFIG_FILE (str): Name of the device configuration file.

Enumerations:
-------------
    FlexAxis: Defines the axes (X, Y, Z) used for device movements.
    FlexPipettes: Specifies available pipettes for the Opentrons Flex device.
    FlexStandardTipRacks: Lists the standard tip racks supported by the device.
    FlexCommandType: Contains commands supported by the device for various operations.
    FlexMountPositions: Represents the mount positions (left or right) for pipettes.
    FlexDeckLocations: Defines deck locations on the device for labware or modules.
    FlexIntents: Specifies high-level intents for device operation.
    FlexActions: Contains operational actions like pause, play, and stop.
    FlexLights: Specifies light states (on/off) for the device.

Classes:
--------
    FlexSlotName:
        Contains utility methods for retrieving slot names based on deck location.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2025-08-07

Version:
    1.0
"""

from enum import Enum
import os

try:
    from QATCH.common.architecture import Architecture
except (ModuleNotFoundError, ImportError):

    class Architecture:
        def get_path():
            return os.path.join("FluxControls")


"""str: Default HTTP port for communication."""
HTTP_PORT = "31950"

"""dict: HTTP headers specifying the Opentrons version."""
HEADERS = {"opentrons-version": "3"}

"""str: Name of the device configuration file."""
DEFAULT_DEV_CONFIG_PATH = os.path.join(
    Architecture.get_path(), "devices", "device_config.json"
)


class Axis(Enum):
    """Enumeration of axes for device movement."""

    X = "x"
    Y = "y"
    Z = "z"


class Pipettes(Enum):
    """Enumeration of pipettes supported by the Opentrons Flex device."""

    EMPTY = "None"
    P50_SINGLE_FLEX = "p50_single_flex"
    P50_MULTI_FLEX = "p50_multi_flex"
    P300_SINGLE_FLEX_GEN2 = "p300_single_gen2"
    P1000_SINGLE_FLEX = "p1000_single_flex"
    P1000_MULTI_FLEX = "p1000_multi_flex"


class FlexStandardTipRacks(Enum):
    """Enumeration of standard tip racks for the device."""

    TR_96_50 = "opentrons_flex_96_tiprack_50ul"
    TR_96_200 = "opentrons_flex_96_tiprack_200ul"
    TR_96_300 = "opentrons_96_tiprack_300ul"
    TR_96_1000 = "opentrons_flex_96_filtertiprack_1000ul"
    FTR_96_50 = "opentrons_flex_96_filtertiprack_50ul"
    FTR_96_200 = "opentrons_flex_96_filtertiprack_200ul"
    FTR_96_300 = "opentrons_96_filtertiprack_300ul"
    FTR_96_1000 = "opentrons_flex_96_filtertiprack_1000ul"


class CommandType(Enum):
    """Enumeration of commands supported by the device."""

    LOAD_PIPETTE = "loadPipette"
    LOAD_LABWARE = "loadLabware"
    PICKUP_TIP = "pickUpTip"
    ASPIRATE = "aspirate"
    DISPENSE = "dispense"
    BLOWOUT = "blowout"
    DROP_TIP = "dropTip"
    MOVE_TO_WELL = "moveToWell"
    MOVE_TO_COORDINATES = "moveToCoordinates"
    MOVE_RELATIVE = "moveRelative"


class MountPositions(Enum):
    """Enumeration of pipette mount positions."""

    LEFT_MOUNT = "left"
    RIGHT_MOUNT = "right"


class DeckLocations(Enum):
    """Enumeration of deck locations on the device."""

    A1 = "A1"
    A2 = "A2"
    A3 = "A3"
    A4 = "A4"
    B1 = "B1"
    B2 = "B2"
    B3 = "B3"
    B4 = "B4"
    C1 = "C1"
    C2 = "C2"
    C3 = "C3"
    C4 = "C4"
    D1 = "D1"
    D2 = "D2"
    D3 = "D3"
    D4 = "D4"


class SlotName:
    """
    Utility class for mapping deck locations to slot names.

    Methods:
    --------
        get_slot_name(location): Retrieves the slot name corresponding to a deck location.
    """

    @staticmethod
    def get_slot_name(location):
        """
        Retrieve the slot name for a given deck location.

        Args:
            location (FlexDeckLocations): The deck location.

        Returns:
            dict: A dictionary containing the slot name, or None if the location is invalid.
        """
        slot_names = {
            DeckLocations.A1: {"slotName": 1},
            DeckLocations.A2: {"slotName": 2},
            DeckLocations.A3: {"slotName": 3},
            DeckLocations.A4: {"slotName": 4},
            DeckLocations.B1: {"slotName": 5},
            DeckLocations.B2: {"slotName": 6},
            DeckLocations.B3: {"slotName": 7},
            DeckLocations.B4: {"slotName": 8},
            DeckLocations.C1: {"slotName": 9},
            DeckLocations.C2: {"slotName": 10},
            DeckLocations.C3: {"slotName": 11},
            DeckLocations.C4: {"slotName": 12},
            DeckLocations.D1: {"slotName": 13},
            DeckLocations.D2: {"slotName": 14},
            DeckLocations.D3: {"slotName": 15},
            DeckLocations.D4: {"slotName": 16},
        }
        return slot_names.get(location)


class Intents(Enum):
    """Enumeration of high-level intents for device operation."""

    SETUP = "setup"


class Actions(Enum):
    """Enumeration of operational actions."""

    PAUSE = "pause"
    PLAY = "play"
    STOP = "stop"
    RESUME = "resume"


class Lights(Enum):
    """Enumeration of light states."""

    ON = {"on": True}
    OFF = {"on": False}
