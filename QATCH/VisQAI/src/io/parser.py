"""
parser.py

This module defines the `Parser` class for extracting formulation and component information
from an XML file. It provides methods to retrieve text values, parameter attributes, and
construct `Protein`, `Buffer`, `Stabilizer`, `Surfactant`, and `Formulation` objects
based on the XML structure. The parser is designed to work with the expected XML schema
that contains a `<params>` section with multiple `<param>` entries.

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2025-10-22

Version:
    1.2
"""
import zipfile
import re
import numpy as np
import os
import xml.etree.ElementTree as ET
from typing import Any, Optional, Type
from pathlib import Path
try:
    class Log:
        @staticmethod
        def d(tag, msg=""): print("DEBUG:", tag, msg)
        @staticmethod
        def i(tag, msg=""): print("INFO:", tag, msg)
        @staticmethod
        def w(tag, msg=""): print("WARNING:", tag, msg)
        @staticmethod
        def e(tag, msg=""): print("ERROR:", tag, msg)
    from src.models.ingredient import (
        Protein, Buffer, Stabilizer, Surfactant, Salt, Excipient
    )
    from src.models.formulation import ViscosityProfile, Formulation
    from src.controller.ingredient_controller import IngredientController
    from src.db.db import Database

except (ModuleNotFoundError, ImportError):
    from QATCH.VisQAI.src.models.ingredient import (
        Protein, Buffer, Stabilizer, Surfactant, Salt, Excipient
    )
    from QATCH.VisQAI.src.models.formulation import ViscosityProfile, Formulation
    from QATCH.common.logger import Logger as Log
    from QATCH.VisQAI.src.controller.ingredient_controller import IngredientController
    from QATCH.VisQAI.src.db.db import Database

TAG = "[Parser]"


class Parser:
    """XML parser for extracting formulation-related parameters and creating model objects.

    The parser expects an XML file with a `<params>` section containing multiple `<param>` elements.
    Each `<param>` must have a `name` attribute and may have a `value` attribute or other attributes
    (e.g., `units`). This class provides methods to retrieve values and build `Protein`, `Buffer`,
    `Stabilizer`, `Surfactant`, and `Formulation` instances based on the XML content.
    """

    def __init__(self, xml_path: str):
        """Initialize the parser by loading and parsing the XML file.

        Args:
            xml_path (str): Filesystem path to the XML file.

        Raises:
            FileNotFoundError: If the file does not exist at `xml_path`.
            ET.ParseError: If the XML is malformed and cannot be parsed.
        """
        xml_path = Path(xml_path)
        base_path = xml_path.parent
        capture_path = next(base_path.glob("capture.zip"), None)
        if not os.path.exists(capture_path):
            raise FileNotFoundError(
                f"capture.zip not found at path `{capture_path}`.")
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML not found at path `{xml_path}`.")

        tree = ET.parse(xml_path)
        self.root = tree.getroot()
        self.profile_shears = [1e2, 1e3, 1e4, 1e5, 15000000]
        params_list = self.root.findall("params")
        if not params_list:
            self.params = None
        else:
            # Use the last <params> section if multiple are present
            self.params = params_list[-1]
        self.database = Database(parse_file_key=True)
        self.ing_ctrl = IngredientController(self.database)

    def get_text(self, elem: ET.Element, tag: str, cast_type: Type) -> Any:
        """Retrieve and cast the text content of a child element.

        Args:
            elem (ET.Element): Parent XML element.
            tag (str): Tag name of the child element to find.
            cast_type (Type): Callable type (e.g., `int`, `float`, `str`) used to cast the text.

        Returns:
            Any: The value of the child element, cast to `cast_type`.

        Raises:
            ValueError: If no child with the specified `tag` is found, or if casting fails.
        """
        txt = elem.findtext(tag)
        if txt is None:
            raise ValueError(f"Missing <{tag}> in <{elem.tag}>")
        try:
            return cast_type(txt)
        except ValueError:
            raise ValueError(f"Cannot cast '{txt}' of <{tag}> to {cast_type}")

    def get_param(
        self,
        name: str,
        cast_type: Type = str,
        required: bool = True
    ) -> Any:
        """Retrieve a parameter value by name from the `<params>` section and cast it.

        Args:
            name (str): The `name` attribute of the `<param>` element to retrieve.
            cast_type (Type, optional): Callable type (e.g., `int`, `float`, `str`) to cast the value.
                Defaults to `str`.
            required (bool, optional): Whether the parameter is required. If True and the parameter
                is missing, raises an error. If False, returns `None` when missing. Defaults to True.

        Returns:
            Any: The parameter value cast to `cast_type`, or `None` if not found and `required` is False.

        Raises:
            ValueError: If `<params>` is missing but `required` is True,
                        or if the `<param>` element with `name` is missing but `required` is True,
                        or if the `value` attribute is missing, or if casting fails.
        """
        if self.params is None:
            if required:
                raise ValueError("No <params> section available")
            return None

        el = self.params.find(f"param[@name='{name}']")
        if el is None:
            if required:
                raise ValueError(f"Missing param name='{name}' in <params>")
            return None

        val = el.get("value")
        if val is None:
            raise ValueError(f"Param '{name}' has no value attribute")
        try:
            return cast_type(val)
        except ValueError:
            raise ValueError(
                f"Cannot cast param '{name}' value '{val}' to {cast_type}")

    def get_param_attr(
        self,
        name: str,
        attr: str,
        required: bool = False
    ) -> Optional[str]:
        """Retrieve a specific attribute from a `<param>` element.

        Args:
            name (str): The `name` attribute of the `<param>` element.
            attr (str): The attribute to retrieve from the `<param>` element.
            required (bool, optional): Whether the attribute is required. If True and the attribute
                is missing, raises an error. If False, returns `None`. Defaults to False.

        Returns:
            Optional[str]: The value of the requested attribute, or `None` if not present and `required` is False.

        Raises:
            ValueError: If `<params>` is missing (when `required` is True),
                        or if the `<param>` element is missing or lacks the specified attribute when `required` is True.
        """
        if self.params is None:
            return None
        el = self.params.find(f"param[@name='{name}']")
        if el is None or attr not in el.attrib:
            if required:
                raise ValueError(f"Param '{name}' missing attribute '{attr}'")
            return None
        return el.get(attr)

    def get_protein(self) -> Protein:
        """Construct and return a `Protein` object with concentration and units from `<params>`.

        Expects the following `<param>` entries in `<params>`:
            - `protein_type` (string)
            - `protein_mw` (float)
            - `protein_pI_mean` (float)
            - `protein_pI_range` (float)
            - `protein_concentration` (float)
            - `units` attribute on `protein_concentration` (string, required)

        Returns:
            Protein: A dictionary-like structure containing:
                {
                    "protein": <Protein instance>,
                    "concentration": <float>,
                    "units": <str>
                }

        Raises:
            ValueError: If any required parameter is missing or cannot be cast.
        """
        name = self.get_param("protein_type", str)
        conc = self.get_param("protein_concentration", float)
        protein_obj = self.ing_ctrl.get_protein_by_name(name)
        if protein_obj is None:
            raise ValueError(
                "Protein could not be fetched from persistent store.")
        molecular_weight = protein_obj.molecular_weight
        pI_mean = protein_obj.pI_mean
        pI_range = protein_obj.pI_range
        units = self.get_param_attr(
            "protein_concentration", "units", required=True)

        return {
            "protein": Protein(
                enc_id=-1,
                name=name,
                molecular_weight=molecular_weight,
                pI_mean=pI_mean,
                pI_range=pI_range
            ),
            "concentration": conc,
            "units": units
        }

    def get_buffer(self) -> Buffer:
        """Construct and return a `Buffer` object with concentration and units from `<params>`.

        Expects the following `<param>` entries in `<params>`:
            - `buffer_type` (string)
            - `buffer_pH` (float)
            - `buffer_concentration` (float)
            - `units` attribute on `buffer_concentration` (string, optional)

        Returns:
            Buffer: A dictionary-like structure containing:
                {
                    "buffer": <Buffer instance>,
                    "concentration": <float>,
                    "units": <str> or None
                }

        Raises:
            ValueError: If any required parameter is missing or cannot be cast.
        """
        name = self.get_param("buffer_type", str)
        conc = self.get_param("buffer_concentration", float)
        units = self.get_param_attr("buffer_concentration", "units")
        buffer_obj = self.ing_ctrl.get_buffer_by_name(name)
        if buffer_obj is None:
            raise ValueError(
                "Buffer could not be fetched from persistent store.")
        else:
            buffer_ph = buffer_obj.pH
        return {
            "buffer": Buffer(enc_id=-1, name=name, pH=buffer_ph),
            "concentration": conc,
            "units": units,
        }

    def get_stabilizer(self) -> Stabilizer:
        """Construct and return a `Stabilizer` object with concentration and units from `<params>`.

        Expects the following `<param>` entries in `<params>`:
            - `stabilizer_type` (string)
            - `stabilizer_concentration` (float)
            - `units` attribute on `stabilizer_concentration` (string, optional)

        Returns:
            Stabilizer: A dictionary-like structure containing:
                {
                    "stabilizer": <Stabilizer instance>,
                    "concentration": <float>,
                    "units": <str> or None
                }

        Raises:
            ValueError: If any required parameter is missing or cannot be cast.
        """
        name = self.get_param("stabilizer_type", str)
        conc = self.get_param("stabilizer_concentration", float)
        units = self.get_param_attr("stabilizer_concentration", "units")

        return {
            "stabilizer": Stabilizer(enc_id=-1, name=name),
            "concentration": conc,
            "units": units
        }

    def get_surfactant(self) -> Surfactant:
        """Construct and return a `Surfactant` object with concentration and units from `<params>`.

        Expects the following `<param>` entries in `<params>`:
            - `surfactant_type` (string)
            - `surfactant_concentration` (float)
            - `units` attribute on `surfactant_concentration` (string, optional)

        Returns:
            Surfactant: A dictionary-like structure containing:
                {
                    "surfactant": <Surfactant instance>,
                    "concentration": <float>,
                    "units": <str> or None
                }

        Raises:
            ValueError: If any required parameter is missing or cannot be cast.
        """
        name = self.get_param("surfactant_type", str)
        conc = self.get_param("surfactant_concentration", float)
        units = self.get_param_attr("surfactant_concentration", "units")

        return {
            "surfactant": Surfactant(enc_id=-1, name=name),
            "concentration": conc,
            "units": units
        }

    def get_excipient(self) -> Excipient:
        """Construct and return a `Excipient` object with concentration and units from `<params>`.
        """
        name = self.get_param("excipient_type", str)
        conc = self.get_param("excipient_concentration", float)
        units = self.get_param_attr("excipient_concentration", "units")

        return {
            "excipient": Excipient(enc_id=-1, name=name),
            "concentration": conc,
            "units": units
        }

    def get_salt(self) -> Salt:
        """Construct and return a `Salt` object with concentration and units from `<params>`.
        """
        name = self.get_param("salt_type", str)
        conc = self.get_param("salt_concentration", float)
        units = self.get_param_attr("salt_concentration", "units")

        return {
            "salt": Salt(enc_id=-1, name=name),
            "concentration": conc,
            "units": units
        }

    def get_viscosity_profile(self) -> dict:
        """
        Locate and extract viscosity data from the largest analyze-[INT].zip file.

        Returns:
            dict: {
                "shear_rate": np.ndarray,
                "viscosity": np.ndarray,
                "temperature": np.ndarray
            }
        """
        base_dir = Path(self.root.base) if hasattr(
            self.root, 'base') else Path(self.root.tag).parent
        capture_zip = next(base_dir.glob("capture.zip"), None)
        if not capture_zip or not capture_zip.exists():
            raise FileNotFoundError(f"capture.zip not found in {base_dir}")

        # Unzip capture.zip to inspect analyze zips
        with zipfile.ZipFile(capture_zip, "r") as capture:
            analyze_zips = [name for name in capture.namelist()
                            if re.match(r".*analyze-\d+\.zip$", name)]

            if not analyze_zips:
                raise FileNotFoundError(
                    "No analyze.zip files found inside capture.zip")

            # Select the one with the largest integer
            largest_zip_name = max(
                analyze_zips, key=lambda n: int(
                    re.search(r"analyze-(\d+)\.zip", n).group(1))
            )

            # Extract the analyze zip in memory
            with capture.open(largest_zip_name) as inner_zip_file:
                with zipfile.ZipFile(inner_zip_file) as analyze_zip:
                    csv_file_name = [n for n in analyze_zip.namelist()
                                     if n.endswith("_analyze_out.csv")]
                    if not csv_file_name:
                        raise FileNotFoundError(
                            "No *_analyze_out.csv found inside analyze zip")

                    with analyze_zip.open(csv_file_name[0]) as csv_f:
                        csv_data = np.loadtxt(
                            (line.decode("utf-8") for line in csv_f),
                            delimiter=",",
                            skiprows=1,
                            usecols=(0, 2, 4)
                        )
                        shear_rate = csv_data[:, 0]
                        viscosity = csv_data[:, 1]
                        temperature = csv_data[:, 2]

        return {
            "shear_rate": shear_rate,
            "viscosity": viscosity,
            "temperature": temperature
        }

    def get_formulation(self,) -> Formulation:
        """Construct a `Formulation` object using parsed parameters and provided arguments.
        Returns:
            Formulation: A fully populated `Formulation` instance.
        Raises:
            TypeError: If `salt` is not a `Salt` instance, or if `vp` dict does not contain
                the required keys with list values.
            ValueError: If any provided concentration or temperature value is invalid.
        """
        formulation = Formulation()

        buffer_data = self.get_buffer()
        protein_data = self.get_protein()
        surfactant_data = self.get_surfactant()
        stabilizer_data = self.get_stabilizer()
        excipient_data = self.get_excipient()
        salt_data = self.get_salt()
        vp_data = self.get_viscosity_profile()

        formulation.set_buffer(
            buffer=buffer_data["buffer"],
            concentration=buffer_data["concentration"],
            units=buffer_data["units"]
        )
        formulation.set_protein(
            protein=protein_data["protein"],
            concentration=protein_data["concentration"],
            units=protein_data["units"]
        )
        formulation.set_surfactant(
            surfactant=surfactant_data["surfactant"],
            concentration=surfactant_data["concentration"],
            units=surfactant_data["units"]
        )
        formulation.set_stabilizer(
            stabilizer=stabilizer_data["stabilizer"],
            concentration=stabilizer_data["concentration"],
            units=stabilizer_data["units"]
        )
        formulation.set_excipient(
            excipient=excipient_data["excipient"],
            concentration=excipient_data["concentration"],
            units=excipient_data["units"]
        )
        formulation.set_salt(
            salt=salt_data["salt"],
            concentration=salt_data["concentration"],
            units=salt_data["units"]
        )
        formulation.set_temperature(temp=vp_data["temperature"])
        viscosity_profile = ViscosityProfile(
            shear_rates=vp_data["shear_rate"],
            viscosities=vp_data["viscosity"],
            units="cP"
        )
        formulation.set_viscosity_profile(profile=viscosity_profile)

        return formulation
