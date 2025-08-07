"""
pipette.py

Module for representing and validating pipette configurations for Opentrons Flex.

Provides the Pipette class to encapsulate pipette type and mounting position,
including validation against defined Pipettes and MountPositions enums.
Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2025-08-07

Version:
    1.0
"""
from typing import Any

try:
    from src.constants import Pipettes, MountPositions

    class Log:
        """Simple logging shim when running standalone."""
        def d(tag, msg=""): print("DEBUG:", tag, msg)
        def i(tag, msg=""): print("INFO:", tag, msg)
        def w(tag, msg=""): print("WARNING:", tag, msg)
        def e(tag, msg=""): print("ERROR:", tag, msg)
    Log.i(print("Running FluxControls as standalone app"))
except (ImportError, ModuleNotFoundError):
    from QATCH.common.logger import Logger as Log
    from QATCH.FluxControls.src.constants import Pipettes, MountPositions


class Pipette:
    """Encapsulates a pipette type and its mount position on the robot.

    Attributes:
        _id (Any): Optional identifier for this pipette instance.
        _pipette (Pipettes): Enum value for pipette type.
        _mount_position (MountPositions): Enum value for mount position.
    """

    def __init__(self, pipette: Pipettes, mount_position: MountPositions) -> None:
        """Initialize a Pipette instance.

        Args:
            pipette (Pipettes): The pipette model/type.
            mount_position (MountPositions): The mount position (e.g., left, right).
        """
        self._id: Any = None
        self._pipette: Pipettes = pipette
        self._mount_position: MountPositions = mount_position

    def _is_valid_pipette(self, pipette: Pipettes) -> bool:
        """Validate that the pipette is a supported enum value.

        Args:
            pipette (Pipettes): Candidate pipette enum.

        Returns:
            bool: True if pipette is valid, False otherwise.
        """
        return pipette in Pipettes

    def _is_valid_mount_position(self, mount_position: MountPositions) -> bool:
        """Validate that the mount position is supported.

        Args:
            mount_position (MountPositions): Candidate mount position enum.

        Returns:
            bool: True if mount position is valid, False otherwise.
        """
        return mount_position in MountPositions

    @property
    def id(self) -> Any:
        """Any: Identifier for this pipette instance."""
        return self._id

    @id.setter
    def id(self, id: Any) -> None:
        """Set the identifier for this pipette.

        Args:
            id (Any): Identifier value.
        """
        self._id = id

    @property
    def pipette(self) -> str:
        """str: The pipette model name as a string."""
        return self._pipette.value

    @pipette.setter
    def pipette(self, pipette: Pipettes) -> None:
        """Set the pipette type, validating against supported models.

        Args:
            pipette (Pipettes): New pipette enum value.

        Raises:
            ValueError: If pipette is not supported.
        """
        if self._is_valid_pipette(pipette=pipette):
            self._pipette = pipette
        else:
            Log.e(f"Invalid pipette tip {pipette.value}.")
            raise ValueError(f"Invalid pipette tip {pipette.value}.")

    @property
    def mount_position(self) -> str:
        """str: The mount position name as a string."""
        return self._mount_position.value

    @mount_position.setter
    def mount_position(self, mount_position: MountPositions) -> None:
        """Set the mount position, validating against supported positions.

        Args:
            mount_position (MountPositions): New mount position enum value.

        Raises:
            ValueError: If mount position is not supported.
        """
        if self._is_valid_mount_position(mount_position=mount_position):
            self._mount_position = mount_position
        else:
            Log.e(f"Invalid mount position {mount_position.value}.")
            raise ValueError(f"Invalid mount position {mount_position.value}.")
