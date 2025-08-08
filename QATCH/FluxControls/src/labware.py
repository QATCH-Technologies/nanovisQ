"""
labware.py

Module for representing and loading custom and standard labware definitions.

This module provides classes to encapsulate labware definitions, including ordering of wells,
brand information, metadata, dimensions, well specifications, grouping, parameters,
and offsets for both custom JSON-based labware and standard labware definitions.

Classes:
    Ordering: Represents well ordering in labware.
    Brand: Brand metadata for labware.
    Metadata: Display metadata for labware.
    Dimensions: Physical dimensions of labware.
    Well: Specification of an individual well.
    Wells: Collection of well specifications.
    GroupMetadata: Metadata for a group of wells.
    Group: Grouping of wells with associated metadata.
    Parameters: Operational parameters and quirks for labware usage.
    CornerOffsetFromSlot: Offset of labware corner from slot origin.
    StackingOffsetWithLabware: Stacking offsets when labware is placed on other labware.
    Labware: High-level representation of labware, supporting JSON and standard definitions.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2025-08-07

Version:
    1.0
"""
from typing import Union, List, Dict, Any
import json
import os

try:
    from src.constants import DeckLocations
    from src.standard_labware import StandardLabware

    class Log:
        """Simple logging shim when running as standalone."""
        def d(tag, msg=""): print("DEBUG:", tag, msg)
        def i(tag, msg=""): print("INFO:", tag, msg)
        def w(tag, msg=""): print("WARNING:", tag, msg)
        def e(tag, msg=""): print("ERROR:", tag, msg)
    Log.i(print("Running FluxControls as standalone app"))
except (ModuleNotFoundError, ImportError):
    from QATCH.FluxControls.src.constants import DeckLocations
    from QATCH.FluxControls.src.standard_labware import StandardLabware
    from QATCH.common.logger import Logger as Log


class Ordering:
    """Represents the ordering of wells in labware.

    Args:
        ordering (List[List[str]]): Nested lists of well identifiers in the defined order.

    Attributes:
        ordering (List[List[str]]): The ordering of well names.
    """

    def __init__(self, ordering: List[List[str]]):
        self._ordering = ordering

    @property
    def ordering(self) -> List[List[str]]:
        """List of lists of well names representing the ordering."""
        return self._ordering


class Brand:
    """Brand metadata for labware.

    Args:
        brand (str): Brand display name.
        brandId (List[str]): Identifiers for the brand.

    Attributes:
        brand (str): The brand name.
        brand_id (List[str]): List of brand identifiers.
    """

    def __init__(self, brand: str, brandId: List[str]):
        self._brand = brand
        self._brand_id = brandId

    @property
    def brand(self) -> str:
        """The brand display name."""
        return self._brand

    @property
    def brand_id(self) -> List[str]:
        """Identifiers for the brand."""
        return self._brand_id


class Metadata:
    """Display metadata for labware.

    Args:
        displayName (str): Human-readable display name.
        displayCategory (str): Category for display grouping.
        displayVolumeUnits (str): Units for volume display.
        tags (List[str]): Associated tags.

    Attributes:
        display_name (str): Display name.
        display_category (str): Display category.
        display_volume_units (str): Volume units.
        tags (List[str]): Tags.
    """

    def __init__(self, displayName: str, displayCategory: str, displayVolumeUnits: str, tags: List[str]):
        self._display_name = displayName
        self._display_category = displayCategory
        self._display_volume_units = displayVolumeUnits
        self._tags = tags

    @property
    def display_name(self) -> str:
        """Human-readable name for labware."""
        return self._display_name

    @property
    def display_category(self) -> str:
        """Category used for organizing labware."""
        return self._display_category

    @property
    def display_volume_units(self) -> str:
        """Units for displaying volume."""
        return self._display_volume_units

    @property
    def tags(self) -> List[str]:
        """List of metadata tags."""
        return self._tags


class Dimensions:
    """Physical dimensions of labware.

    Args:
        xDimension (float): X dimension in millimeters.
        yDimension (float): Y dimension in millimeters.
        zDimension (float): Z dimension in millimeters.

    Attributes:
        x_dimension (float): X dimension.
        y_dimension (float): Y dimension.
        z_dimension (float): Z dimension.
    """

    def __init__(self, xDimension: float, yDimension: float, zDimension: float):
        self._x_dimension = xDimension
        self._y_dimension = yDimension
        self._z_dimension = zDimension

    @property
    def x_dimension(self) -> float:
        """The X dimension in millimeters."""
        return self._x_dimension

    @property
    def y_dimension(self) -> float:
        """The Y dimension in millimeters."""
        return self._y_dimension

    @property
    def z_dimension(self) -> float:
        """The Z dimension in millimeters."""
        return self._z_dimension


class Well:
    """Specification of an individual well within labware.

    Args:
        depth (float): Depth of the well.
        totalLiquidVolume (float): Maximum liquid volume.
        shape (str): Shape descriptor, e.g., 'circular'.
        diameter (float): Diameter if circular.
        x (float): X-coordinate of well center.
        y (float): Y-coordinate of well center.
        z (float): Z-coordinate of well bottom.

    Attributes:
        depth (float): Well depth.
        total_liquid_volume (float): Maximum volume.
        shape (str): Shape.
        diameter (float): Diameter.
        x (float): X-coordinate.
        y (float): Y-coordinate.
        z (float): Z-coordinate.
    """

    def __init__(self, depth: float,
                 totalLiquidVolume: float,
                 shape: str,
                 diameter: float,
                 x: float,
                 y: float,
                 z: float):
        self._depth = depth
        self._total_liquid_volume = totalLiquidVolume
        self._shape = shape
        self._diameter = diameter
        self._x = x
        self._y = y
        self._z = z

    @property
    def depth(self) -> float:
        """Depth of the well."""
        return self._depth

    @property
    def total_liquid_volume(self) -> float:
        """Maximum liquid volume of the well."""
        return self._total_liquid_volume

    @property
    def shape(self) -> str:
        """Shape descriptor of the well."""
        return self._shape

    @property
    def diameter(self) -> float:
        """Diameter of the well (for circular wells)."""
        return self._diameter

    @property
    def x(self) -> float:
        """X-coordinate of the well center."""
        return self._x

    @property
    def y(self) -> float:
        """Y-coordinate of the well center."""
        return self._y

    @property
    def z(self) -> float:
        """Z-coordinate of the well bottom."""
        return self._z


class Wells:
    """Container for all wells in labware.

    Args:
        wells (Dict[str, Any]): Mapping of well names to specification dicts.

    Attributes:
        wells (Dict[str, Well]): Mapping of well names to Well objects.
    """

    def __init__(self, wells: Dict[str, Any]):
        self._wells = {key: Well(**value) for key, value in wells.items()}

    @property
    def wells(self) -> Dict[str, Any]:
        """Mapping of well names to Well instances."""
        return self._wells


class GroupMetadata:
    """Metadata for a group of wells.

    Args:
        wellBottomShape (str): Shape of the well bottom.

    Attributes:
        well_bottom_shape (str): Shape of the well bottom.
    """

    def __init__(self, wellBottomShape: str):
        self._well_bottom_shape = wellBottomShape

    @property
    def well_bottom_shape(self) -> str:
        """Shape descriptor for the bottom of wells in this group."""
        return self._well_bottom_shape


class Group:
    """Grouping of wells with associated metadata.

    Args:
        metadata (Dict[str, str]): Metadata for the group.
        wells (List[str]): List of well names in the group.

    Attributes:
        metadata (GroupMetadata): Metadata object for the group.
        wells (List[str]): Well names in the group.
    """

    def __init__(self, metadata: Dict[str, str], wells: List[str]):
        self._metadata = GroupMetadata(**metadata)
        self._wells = wells

    @property
    def metadata(self) -> GroupMetadata:
        """Metadata for this well group."""
        return self._metadata

    @property
    def wells(self) -> List[str]:
        """List of well names that belong to this group."""
        return self._wells


class Parameters:
    """Operational parameters and quirks for labware usage.

    Args:
        format (str): Load format descriptor.
        quirks (List[str]): Any special quirks.
        isTiprack (bool): Whether this labware is a tiprack.
        isMagneticModuleCompatible (bool): Compatibility flag.
        loadName (str): Load name identifier.

    Attributes:
        format (str): Load format.
        quirks (List[str]): Quirks.
        is_tiprack (bool): Tiprack flag.
        is_magnetic_module_compatible (bool): Compatibility flag.
        load_name (str): Load name.
    """

    def __init__(
        self,
        format: str,
        quirks: List[str],
        isTiprack: bool,
        isMagneticModuleCompatible: bool,
        loadName: str
    ):
        self._format = format
        self._quirks = quirks
        self._is_tiprack = isTiprack
        self._is_magnetic_module_compatible = isMagneticModuleCompatible
        self._load_name = loadName

    @property
    def format(self) -> str:
        """Format descriptor used for loading."""
        return self._format

    @property
    def quirks(self) -> List[str]:
        """List of special quirks for this labware."""
        return self._quirks

    @property
    def is_tiprack(self) -> bool:
        """Whether the labware is a tiprack."""
        return self._is_tiprack

    @property
    def is_magnetic_module_compatible(self) -> bool:
        """Compatibility with a magnetic module."""
        return self._is_magnetic_module_compatible

    @property
    def load_name(self) -> str:
        """Identifier used for loading the labware."""
        return self._load_name


class CornerOffsetFromSlot:
    """Offset of the labware corner from the slot origin.

    Args:
        x (float): X offset.
        y (float): Y offset.
        z (float): Z offset.

    Attributes:
        x (float): X offset.
        y (float): Y offset.
        z (float): Z offset.
    """

    def __init__(self, x: float, y: float, z: float):
        self._x = x
        self._y = y
        self._z = z

    def get_offsets(self) -> dict:
        """Get a dict of the corner offsets.

        Returns:
            dict: Offsets with keys 'x', 'y', 'z'.
        """
        return {"x": self.x, "y": self.y, "z": self.z}

    @property
    def x(self) -> float:
        """X-axis offset."""
        return self._x

    @property
    def y(self) -> float:
        """Y-axis offset."""
        return self._y

    @property
    def z(self) -> float:
        """Z-axis offset."""
        return self._z


class StackingOffsetWithLabware:
    """Stacking offsets when this labware is placed on another labware.

    Args:
        stackingOffsetWithLabware (Dict[str, Dict[str, float]]): Mapping of labware names to offsets.

    Attributes:
        stacking_offset_with_labware (Dict[str, Dict[str, float]]): Offsets mapping.
    """

    def __init__(self, stackingOffsetWithLabware: Dict[str, Dict[str, float]]):
        self._stacking_offset_with_labware = stackingOffsetWithLabware

    @property
    def stacking_offset_with_labware(self) -> Dict[str, Dict[str, float]]:
        """Offsets for stacking with other labware."""
        return self._stacking_offset_with_labware


class Labware:
    """High-level representation of labware definition and usage.

    Supports loading from JSON definitions or standard labware enums.

    Args:
        location (DeckLocations): Deck slot location.
        labware_definition (Union[str, StandardLabware]): File path to custom JSON or a StandardLabware instance.

    Attributes:
        data (Dict[str, Any]): Raw JSON data if loaded from file.
        ordering (Ordering): Well ordering.
        brand (Brand): Brand metadata.
        metadata (Metadata): Display metadata.
        dimensions (Dimensions): Physical dimensions.
        wells (Wells): Well specifications.
        groups (List[Group]): Well groupings.
        parameters (Parameters): Operational parameters.
        corner_offset_from_slot (CornerOffsetFromSlot): Corner offsets.
        stacking_offset_with_labware (StackingOffsetWithLabware): Stacking offsets.
        location (DeckLocations): Deck location.
        display_name (str): Display name.
        load_name (str): Load identifier.
        name_space (str): Namespace for definition.
        version (int): Definition version.
        is_tiprack (bool): Tiprack flag.
    """

    def __init__(
        self,
        location: DeckLocations,
        labware_definition: Union[str, StandardLabware]
    ):
        if isinstance(labware_definition, str) and os.path.isfile(labware_definition):
            Log.i(
                f"Loading custom labware definition from: {labware_definition} @ {location.value}")
            self.data = self.load_json(labware_definition)
            self.ordering = Ordering(self.data["ordering"])
            self.brand = Brand(**self.data["brand"])
            self.metadata = Metadata(**self.data["metadata"])
            self.dimensions = Dimensions(**self.data["dimensions"])
            self.wells = Wells(self.data["wells"])
            self.groups = [Group(**g) for g in self.data["groups"]]
            self.parameters = Parameters(**self.data["parameters"])
            self.schema_version = self.data["schemaVersion"]
            self.corner_offset_from_slot = CornerOffsetFromSlot(
                **self.data["cornerOffsetFromSlot"]
            )
            self.stacking_offset_with_labware = StackingOffsetWithLabware(
                self.data["stackingOffsetWithLabware"]
            )
            # TODO: (8/8) Figure out what the labware ID is.  I think Opentrons assigns this ID
            # on the Flex once it's uploaded but I do not remeber.
            self.id = None

            self.location = location
            self.display_name = self.metadata.display_name
            self.load_name = self.parameters.load_name
            self.name_space = self.data["namespace"]
            self.version = self.data["version"]
            self.is_tiprack = self.parameters.is_tiprack

        elif isinstance(labware_definition, StandardLabware):
            Log.i(
                f"Loading standard labware definition {labware_definition.get_display_name()} @ {location.value}")
            # TODO: (8/8) Figure out what the labware ID is.  I think Opentrons assigns this ID
            # on the Flex once it's uploaded but I do not remeber.
            self.id = None
            self.location = location
            self.display_name = labware_definition.get_display_name()
            self.load_name = labware_definition.get_load_name()
            self.name_space = labware_definition.get_name_space()
            self.version = labware_definition.get_version()
            self.is_tiprack = labware_definition.is_tiprack()

        else:
            Log.e("Invalid labware definition type: %s",
                  type(labware_definition))
            raise ValueError(
                "labware_definition must be either a file path (str) or a standard labware enum value."
            )

    def get_offsets(self) -> dict:
        """Get the corner offsets for this labware.

        Returns:
            dict: Offsets with keys 'x', 'y', 'z'.
        """
        return self.corner_offset_from_slot.get_offsets()

    @property
    def id(self) -> int:
        """Identifier for this labware instance."""
        return self._id

    @id.setter
    def id(self, value: int) -> None:
        """Set identifier for this labware instance."""
        self._id = value

    @property
    def location(self) -> DeckLocations:
        """Deck slot location for this labware."""
        return self._location

    @location.setter
    def location(self, loc: DeckLocations) -> None:
        """Set deck slot location."""
        self._location = loc

    @property
    def display_name(self) -> str:
        """Display name of the labware."""
        return self._display_name

    @display_name.setter
    def display_name(self, name: str) -> None:
        """Set display name for the labware."""
        self._display_name = name

    @property
    def load_name(self) -> str:
        """Load name identifier of the labware."""
        return self._load_name

    @load_name.setter
    def load_name(self, name: str) -> None:
        """Set load name identifier."""
        self._load_name = name

    @property
    def name_space(self) -> str:
        """Namespace for the labware definition."""
        return self._name_space

    @name_space.setter
    def name_space(self, ns: str) -> None:
        """Set namespace for the labware definition."""
        self._name_space = ns

    @property
    def version(self) -> int:
        """Version number of the labware definition."""
        return self._version

    @version.setter
    def version(self, ver: int) -> None:
        """Set version number."""
        self._version = ver

    @property
    def is_tiprack(self) -> bool:
        """Whether this labware functions as a tiprack."""
        return self._is_tiprack

    @is_tiprack.setter
    def is_tiprack(self, tip: bool) -> None:
        """Set tiprack flag."""
        self._is_tiprack = tip

    @staticmethod
    def load_json(file_path: str) -> Dict[str, Any]:
        """Load and parse a JSON file.

        Args:
            file_path (str): Path to the JSON file.

        Returns:
            Dict[str, Any]: Parsed JSON content.
        """
        with open(file_path, 'r') as f:
            return json.load(f)
