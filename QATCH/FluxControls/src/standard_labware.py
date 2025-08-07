"""
standard_labware.py

Module defining standardized labware types for Opentrons Flex.

This module provides an enumeration of all built-in labware definitions grouped by category,
such as reservoirs, wellplates, tipracks, tube racks, aluminum blocks, and adapters.
Utilities on the base enum class allow querying properties like tiprack status,
namespace, load name, display name, and version.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2025-08-07

Version:
    1.0
"""
from enum import Enum


class StandardLabware(Enum):
    """Base enum class for standard labware definitions.

    Provides utility methods common to all labware definitions, such as
    identifying tipracks, retrieving version, namespace, load name, and display name.
    """

    def is_tiprack(self) -> bool:
        """Check if this labware functions as a tiprack based on its name.

        Returns:
            bool: True if 'tiprack' is in the definition name, False otherwise.
        """
        return "tiprack" in self.value.lower()

    def get_version(self) -> int:
        """Get the version number for this labware definition.

        Returns:
            int: Version (currently constant 1 for all standard labware).
        """
        return 1

    def get_name_space(self) -> str:
        """Get the namespace under which this labware is defined.

        Returns:
            str: Namespace (currently 'opentrons').
        """
        return "opentrons"

    def get_load_name(self) -> str:
        """Get the load name identifier for this labware.

        Returns:
            str: The enum value string used to load the labware.
        """
        return self.value

    def get_display_name(self) -> str:
        """Get a human-friendly display name for this labware.

        Returns:
            str: Capitalized words from the load name, separated by spaces.
        """
        return ' '.join([word.capitalize() for word in self.value.split('_')])


class StandardReservoirs(StandardLabware):
    """Built-in reservoir definitions (various volumes).

    Enum values represent reservoir models provided by different manufacturers.
    """
    AGILENT_1_RESERVOIR_290ML = "agilent_1_reservoir_290ml"
    AXYGEN_1_RESERVOIR_90ML = "axygen_1_reservoir_90ml"
    NEST_12_RESERVOIR_15ML = "nest_12_reservoir_15ml"
    NEST_1_RESERVOIR_195ML = "nest_1_reservoir_195ml"
    NEST_1_RESERVOIR_290ML = "nest_1_reservoir_290ml"
    USA_SCIENTIFIC_12_RESERVOIR_22ML = "usascientific_12_reservoir_22ml"


class StandardWellplates(StandardLabware):
    """Built-in wellplate definitions (96-, 384-, etc., various volumes).

    Enum values cover common wellplate formats used in PCR and liquid handling.
    """
    APPLIED_BIOSYSTEMS_MICROAMP_384_WELLPLATE_40UL = "appliedbiosystemsmicroamp_384_wellplate_40ul"
    BIORAD_348_WELLPLATE_50UL = 'biorad_384_wellplate_50ul'
    BIORAD_96_WELLPLATE_200UL_PCR = 'biorad_96_wellplate_200ul_pcr'
    CORNING_12_WELLPLATE_6_9ML_FLAT = "corning_12_wellplate_6.9ml_flat"
    CORNING_24_WELLPLATE_3_4ML_FLAT = "corning_24_wellplate_3.4ml_flat"
    CORNING_384_WELLPLATE_122UL_FLAT = "corning_384_wellplate_112ul_flat"
    CORNING_48_WELLPALTGE_1_6ML_FLAT = "corning_48_wellplate_1.6ml_flat"
    CORNING_6_WELLPALTE_16_8ML_FLAT = "corning_6_wellplate_16.8ml_flat"
    CORNING_96_WELLPLATE_360UL_FLAT = "corning_96_wellplate_360ul_flat"
    NEST_96_WELLPLATE_100UL_PCR_FULL_SKIRT = "nest_96_wellplate_100ul_pcr_full_skirt"
    NEST_96_WELLPLATE_200UL_FLAT = "nest_96_wellplate_200ul_flat"
    NEST_96_WELLPLATE_2ML_DEEP = "nest_96_wellplate_2ml_deep"
    OPENTRONS_96_WELLPLATE_200UL_PCR_FULL_SKIRT = "opentrons_96_wellplate_200ul_pcr_full_skirt"
    THERMOSCIENTIFICNUNC_96_WELLPLATE_1300UL = "thermoscientificnunc_96_wellplate_1300ul"
    THERMOSCIENTIFICNUNC_96_WELLPLATE_2000UL = "thermoscientificnunc_96_wellplate_2000ul"
    USASCIENTIFIC_96_wELLPLATE_2_4ML_DEEP = 'usascientific_96_wellplate_2.4ml_deep'


class StandardTipracks(StandardLabware):
    """Built-in tiprack definitions (10µL, 1000µL, etc.).
    """
    GEB_96_TIPRACK_1000UL = 'geb_96_tiprack_1000ul'
    GEB_96_TIPRACK_10UL = 'geb_96_tiprack_10ul'


class StandardTubeRackS(StandardLabware):
    """Built-in tube rack definitions for 50mL, 15mL, 2mL, etc.
    """
    OPENTRONS_10_TUBERACK_FALCON_4X50ML_6X15ML_CONICAL = "opentrons_10_tuberack_falcon_4x50ml_6x15ml_conical"
    OPENTRONS_10_TUBERACK_NEST_4X50ML_6X15ML_CONICAL = "opentrons_10_tuberack_nest_4x50ml_6x15ml_conical"
    OPENTRONS_15_TUBERACK_FALCON_15ML_CONICAL = "opentrons_15_tuberack_falcon_15ml_conical"
    OPENTRONS_15_TUBERACK_NEST_15ML_CONICAL = "opentrons_15_tuberack_nest_15ml_conical"
    OPENTRONS_24_TUBERACK_EPPENDORF_1_5ML_SAFELOCK_SNAPCAP = "opentrons_24_tuberack_eppendorf_1.5ml_safelock_snapcap"
    OPENTRONS_24_TUBERACK_EPPENDORF_2ML_SAFELOCK_SNAPCAP = "opentrons_24_tuberack_eppendorf_2ml_safelock_snapcap"
    OPENTRONS_24_TUBERACK_GENERIC_2ML_SCREWCAP = "opentrons_24_tuberack_generic_2ml_screwcap"
    OPENTRONS_24_TUBERACK_NEST_0_5ML_SCREWCAP = "opentrons_24_tuberack_nest_0.5ml_screwcap"
    OPENTRONS_24_TUBERACK_NEST_1_5ML_SCREWCAP = "opentrons_24_tuberack_nest_1.5ml_screwcap"
    OPENTRONS_24_TUBERACK_NEST_1_5ML_SNAPCAP = "opentrons_24_tuberack_nest_1.5ml_snapcap"
    OPENTRONS_24_TUBERACK_NEST_2ML_SCREWCAP = "opentrons_24_tuberack_nest_2ml_screwcap"
    OPENTRONS_24_TUBERACK_NEST_2ML_SNAPCAP = "opentrons_24_tuberack_nest_2ml_snapcap"
    OPENTRONS_6_TUBERACK_FALCON_50ML_CONICAL = "opentrons_6_tuberack_falcon_50ml_conical"
    OPENTRONS_6_TUBERACK_NEST_50ML_CONICAL = "opentrons_6_tuberack_nest_50ml_conical"


class StandardAluminumBlocks(StandardLabware):
    """Built-in aluminum block definitions for tube and wellplate formats."""
    OPENTRONS_24_ALUMINUMBLOCK_GENERIC_2ML_SCREWCAP = "opentrons_24_aluminumblock_generic_2ml_screwcap"
    OPENTRONS_24_ALUMINUMBLOCK_GENERIC_0_5ML_SCREWCAP = "opentrons_24_aluminumblock_nest_0.5ml_screwcap"
    OPENTRONS_24_ALUMINUMBLOCK_NEST_1_5ML_SCREWCAP = "opentrons_24_aluminumblock_nest_1.5ml_screwcap"
    OPENTRONS_24_ALUMINUMBLOCK_NEST_1_5ML_SNAPCAP = "opentrons_24_aluminumblock_nest_1.5ml_snapcap"
    OPENTRONS_24_ALUMINUMBLOCK_NEST_2ML_SCREWCAP = "opentrons_24_aluminumblock_nest_2ml_screwcap"
    OPENTRONS_24_ALUMINUMBLOCK_NEST_2ML_SNAPCAP = "opentrons_24_aluminumblock_nest_2ml_snapcap"
    OPENTRONS_24_ALUMINUMBLOC_BIORAD_WELLPLATE_200UL = "opentrons_96_aluminumblock_biorad_wellplate_200ul"
    OPENTRONS_96_ALUMINUMBLOCK_GENERIC_PCR_STRIP_200UL = "opentrons_96_aluminumblock_generic_pcr_strip_200ul"
    OPENTRONS_96_ALUMINUMBLOCK_NEST_WELLPLATE_100UL = "opentrons_96_aluminumblock_nest_wellplate_100ul"
    OPENTRONS_96_WELL_ALUMINUM_BLOCK = "opentrons_96_well_aluminum_block"
    OPENTRONS_ALUMINUM_FLAT_BOTTOM_PLATE = "opentrons_aluminum_flat_bottom_plate"


class StandardAdapters(StandardLabware):
    """Built-in adapters definitions for wellplates."""
    OPENTRONS_96_DEEP_WELL_TEMP_MOD_ADAPTER = "opentrons_96_deep_well_temp_mod_adapter"
    OPENTRONS_96_DEEP_WELL_ADAPTER = "opentrons_96_deep_well_adapter"
    OPENTRONS_96_DEEP_WELL_ADAPTER_NEST_WELLPLATE_2ML_DEEP = "opentrons_96_deep_well_adapter_nest_wellplate_2ml_deep"
    OPENTRONS_96_FLAT_BOTTOM_ADAPTER = "opentrons_96_flat_bottom_adapter"
    OPENTRONS_96_FLAT_BOTTOM_ADAPTER_NEST_WELLPLATE_200UL_FLAT = "opentrons_96_flat_bottom_adapter_nest_wellplate_200ul_flat"
    OPENTRONS_96_PCR_ADAPTER = "opentrons_96_pcr_adapter"
    OPENTRONS_96_PCR_ADAPTER_NEST_WELLPLATE_100UL_PCR_FULL_SKIRT = "opentrons_96_pcr_adapter_nest_wellplate_100ul_pcr_full_skirt"
    OPENTRONS_FLEX_96_TIPRACK_ADAPTER = "opentrons_flex_96_tiprack_adapter"
    OPENTRONS_UNIVERSAL_FLAT_ADAPTER = "opentrons_universal_flat_adapter"
    OPENTRONS_UNIVERSAL_FLAT_ADAPTER_CORNING_384_WELLPLATE_112UL_FLAT = "opentrons_universal_flat_adapter_corning_384_wellplate_112ul_flat"
