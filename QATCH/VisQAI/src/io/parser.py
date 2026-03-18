"""
parser.py

This module defines the `Parser` class for extracting formulation and component information
from an XML file. It provides methods to retrieve text values, parameter attributes, and
construct `Protein`, `Buffer`, `Stabilizer`, `Surfactant`, and `Formulation` objects
based on the XML structure. The parser is designed to work with the expected XML schema
that contains a `<params>` section with multiple `<param>` entries.

Author(s):
    Paul MacNichol (paul.macnichol@qatchtech.com)
    Alexander J. Ross (alexander.ross@qatchtech.com)


Date:
    2026-03-16

Version:
    1.7
"""

import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, List, Optional, Type, Union

import numpy as np

try:
    TAG = "[Parser (HEADLESS)]"

    class Log:
        @staticmethod
        def d(TAG, msg=""):
            print("DEBUG:", TAG, msg)

        @staticmethod
        def i(TAG, msg=""):
            print("INFO:", TAG, msg)

        @staticmethod
        def w(TAG, msg=""):
            print("WARNING:", TAG, msg)

        @staticmethod
        def e(TAG, msg=""):
            print("ERROR:", TAG, msg)

    from src.controller.ingredient_controller import IngredientController
    from src.db.db import Database
    from src.io.file_storage import SecureOpen
    from src.models.formulation import Formulation, ViscosityProfile
    from src.models.ingredient import (
        Buffer,
        Excipient,
        Protein,
        Salt,
        Stabilizer,
        Surfactant,
    )

except (ModuleNotFoundError, ImportError):
    TAG = "[Parser]"
    from QATCH.common.logger import Logger as Log
    from QATCH.VisQAI.src.controller.ingredient_controller import IngredientController
    from QATCH.VisQAI.src.db.db import Database
    from QATCH.VisQAI.src.io.file_storage import SecureOpen
    from QATCH.VisQAI.src.models.formulation import Formulation, ViscosityProfile
    from QATCH.VisQAI.src.models.ingredient import (
        Buffer,
        Excipient,
        Protein,
        Salt,
        Stabilizer,
        Surfactant,
    )


class Parser:
    """A domain-specific XML parser for bioformulation data extraction.

    This class provides a comprehensive interface for navigating and extracting
    biochemical run data from XML files and associated filesystem archives. It
    facilitates the conversion of raw XML tags into structured Python objects
    (e.g., Protein, Buffer, Formulation) by combining XML parsing with database
    lookups via an internal IngredientController.

    The parser specifically targets XML structures containing:
        - <run_info>: For identifying run metadata.
        - <params>: For ingredient concentrations and types.
        - <metrics>: For run signatures and measurement results.
        - <audits>: For tracking user actions and timestamps.

    Attributes:
        database (Database): Connection to the backend database for ingredient lookups.
        ing_ctrl (IngredientController): Logic layer for retrieving detailed
            ingredient properties from the database.
        profile_shears (List[float]): A predefined set of shear rates used for
            viscosity interpolation.
        xml_path (Optional[Path]): The path to the currently active XML file.
        base_path (Optional[Path]): The directory containing the active XML,
            used to find sibling data archives (e.g., analyze-*.zip).
        root (Optional[Element]): The root element of the currently loaded XML tree.
        params (Optional[Element]): The specific <params> block currently
            targeted for extraction.
    """

    TAG = "[Parser]"

    def __init__(self, xml_path: str = ""):
        """Initializes the Parser with database connections and default state.

        Sets up the required controllers for ingredient validation and defines
        the standard shear rate profile for viscosity data. While an 'xml_path'
        can be provided for immediate initialization, the intended use case is
        to instantiate the parser and then use the `parse()` method for batch
        or single-file processing.

        Args:
            xml_path (Optional[str]): Path to an XML file or 'capture.zip' file
                to load immediately upon initialization. If passed a ZIP file,
                a matching XML file will be discovered (if exists); otherwise,
                a `FileNotFoundError` will be raised. Defaults to empty str.

        Raises:
            FileNotFoundError: If XML file does not exist at the base ZIP path.
        """
        self.database = Database(parse_file_key=True)
        self.ing_ctrl = IngredientController(self.database)
        self.profile_shears = [1e2, 1e3, 1e4, 1e5, 15000000]
        self.xml_path = None
        self.base_path = None
        self.root = None
        self.params = None

        if xml_path:
            self._load_state(xml_path)

    def parse(self, source: Union[str, List[str]]) -> List[Formulation]:
        """Parses one or more directories or files to extract Formulation objects.

        This method acts as the primary interface for the parser. It identifies
        potential XML files from the provided source(s), recursively searching
        directories if necessary. Each candidate file is loaded and checked for
        the 'bioformulation' flag; only those explicitly marked as such are
        processed. Errors encountered during the parsing of individual files are
        logged, but do not halt the overall batch process.

        Args:
            source: A single path string or a list of path strings. These can
                point directly to XML files or to directories that will be
                searched recursively for XML candidates.

        Returns:
            List[Formulation]: A list of valid, successfully parsed Formulation
                objects extracted from the source(s). Returns an empty list if
                no valid bioformulations are found.

        Note:
            Non-bioformulation XML files and malformed XMLs are skipped and
            noted in the logs.
        """
        if isinstance(source, str):
            sources = [source]
        else:
            sources = source

        formulations = []
        xml_candidates = []
        for path_str in sources:
            path = Path(path_str)
            if not path.exists():
                Log.w(TAG, f"Source path does not exist: {path}")
                continue
            if path.is_file() and path.suffix.lower() == ".xml":
                xml_candidates.append(path)
            elif path.is_dir():

                for root, _, files in os.walk(path):
                    for file in files:
                        if file.lower().endswith(".xml"):
                            xml_candidates.append(Path(root) / file)
        for xml_file in xml_candidates:
            try:
                self._load_state(str(xml_file))
                if not self.is_bioformulation():
                    Log.i(
                        TAG, f"Skipping non-bioformulation file: {xml_file.name}")
                    continue
                form = self.get_formulation()
                if form:
                    formulations.append(form)
            except Exception as e:
                Log.w(TAG, f"Skipping file {xml_file.name}: {e}")
                continue

        return formulations

    def _load_state(self, xml_path_str: str):
        """Initializes the parser state by loading an XML file and setting up core elements.

        This internal method parses the provided XML file, identifies the root
        element, and locates the parameter definitions. It also establishes the
        `base_path` based on the XML's location, which is required for sibling
        file lookups (such as viscosity zip archives). If multiple `<params>`
        sections exist in the XML, only the last one is stored as the active state.

        Args:
            xml_path_str: The file path to the XML run data as a string.

        Raises:
            FileNotFoundError: If the XML file does not exist at the specified path.
            xml.etree.ElementTree.ParseError: If the XML file is malformed or
                cannot be parsed.

        Attributes Set:
            base_path (Path): The directory containing the XML file.
            xml_path (Path): The path to the XML file itself.
            root (Element): The root element of the parsed XML tree.
            params (Optional[Element]): The last `<params>` element found in
                the XML, or None if no parameters are defined.
        """
        input_path = Path(xml_path_str)
        self.base_path = input_path.parent
        self.xml_path = input_path

        if self.xml_path and self.xml_path.suffix != ".xml":
            self.xml_path = next(self.base_path.glob("*.xml"), None)
            if not self.xml_path:
                raise FileNotFoundError(
                    f"XML not found in folder `{self.base_path}`.")

        if not self.xml_path.exists():
            raise FileNotFoundError(
                f"XML not found at path `{self.xml_path}`.")
        tree = ET.parse(self.xml_path)
        self.root = tree.getroot()

        params_list = self.root.findall("params")
        if not params_list:
            self.params = None
        else:
            self.params = params_list[-1]

    def get_param(self, name: str, cast_type: Type = str, required: bool = True) -> Any:
        """Retrieves and casts a parameter value from the `<params>` XML section.

        This method locates a `<param>` element by its 'name' attribute using an
        XPath-style search. If found, it extracts the 'value' attribute and
        attempts to cast it to the specified Python type (e.g., int, float, bool).

        Args:
            name: The 'name' attribute of the `<param>` element to locate.
            cast_type: The Python type to which the parameter value should
                be cast. Defaults to str.
            required: If True, the method raises a ValueError if the `<params>`
                section is missing or the specific parameter name is not found.
                Defaults to True.

        Returns:
            Any: The casted value of the parameter. Returns None if `required`
                is False and the section or parameter is missing.

        Raises:
            ValueError: If `required` is True and the section or parameter is
                missing. Also raised if the 'value' attribute is missing from
                the element, or if the value cannot be cast to `cast_type`.
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
        if len(val.strip()) == 0:
            if required:
                raise ValueError(f"Param '{name}' has empty value attribute")
            return None

        try:
            return cast_type(val)
        except ValueError:
            raise ValueError(
                f"Cannot cast param '{name}' value '{val}' to {cast_type}")

    def get_param_attr(
        self, name: str, attr: str, required: bool = False
    ) -> Optional[str]:
        """Retrieves a specific attribute from a `<param>` element.

        This method searches the internal parameters collection for a `<param>`
        tag with a matching 'name' attribute. If found, it attempts to extract
        the value of the specified attribute (e.g., 'units' or 'id').

        Args:
            name: The value of the 'name' attribute used to identify the
                correct `<param>` element.
            attr: The name of the specific attribute to retrieve from
                the element.
            required: If True, the method raises a ValueError if the
                parameter or the attribute is not found. Defaults to False.

        Returns:
            Optional[str]: The value of the requested attribute if it exists;
                None if the parameter or attribute is missing and not required.

        Raises:
            ValueError: If `required` is True and the specified parameter
                is missing or does not contain the requested attribute.
        """
        if self.params is None:
            return None
        el = self.params.find(f"param[@name='{name}']")
        if el is None or attr not in el.attrib:
            if required:
                raise ValueError(f"Param '{name}' missing attribute '{attr}'")
            return None
        return el.get(attr)

    def get_protein(self) -> dict:
        """Constructs and returns protein-related data using XML params and DB lookups.

        This method attempts to retrieve 'protein_type' and 'protein_concentration'
        from the run XML. It queries the ingredient controller to fetch physical
        properties (molecular weight, pI mean, pI range, and class type) from the
        database. If the protein is unknown to the database or parameters are
        missing, it returns a placeholder Protein object with zeroed values to
        necessitate user intervention.

        Returns:
            dict: A dictionary containing:
                - 'protein' (Protein): A Protein object populated with database
                  attributes or zeroed placeholders.
                - 'concentration' (float): The protein concentration value.
                - 'units' (str): The measurement units (defaults to "mg/mL").
                - 'found' (dict): A mapping of boolean flags indicating if 'name',
                  'conc', and 'units' were explicitly parsed from the XML.

        Notes:
            'protein_type' is a required parameter. If it cannot be found, a
            warning is logged and a default dictionary with 'found' flags set
            to False is returned.
        """
        try:
            name = self.get_param("protein_type", str, required=True)
        except Exception:
            Log.w(
                TAG,
                "<protein_type> param could not be found in run XML; returning default.",
            )
            return {
                "protein": Protein(
                    enc_id=-1,
                    name="None",
                    molecular_weight=0.0,
                    pI_mean=0.0,
                    pI_range=0.0,
                ),
                "concentration": 0.0,
                "units": "mg/mL",
                "found": {
                    "name": False,
                    "conc": False,
                    "units": False,
                },
            }

        protein_obj = self.ing_ctrl.get_protein_by_name(name)
        if protein_obj is None:
            molecular_weight = 0.0
            pI_mean = 0.0
            pI_range = 0.0
            class_type = None
        else:
            molecular_weight = protein_obj.molecular_weight
            pI_mean = protein_obj.pI_mean
            pI_range = protein_obj.pI_range
            class_type = protein_obj.class_type

        try:
            conc = self.get_param("protein_concentration", float)
        except Exception:
            conc = 0.0

        units = self.get_param_attr(
            "protein_concentration", "units", required=False)
        if units is None:
            units = "mg/mL"

        found = {
            "name": name is not None,
            "conc": conc is not None,
            "units": units is not None,
        }

        return {
            "protein": Protein(
                enc_id=-1,
                name=name,
                molecular_weight=molecular_weight,
                pI_mean=pI_mean,
                pI_range=pI_range,
                class_type=class_type,
            ),
            "concentration": conc,
            "units": units,
            "found": found,
        }

    def get_buffer(self) -> dict:
        """Constructs and returns buffer-related data using XML params and DB lookups.

        This method attempts to retrieve the 'buffer_type' and 'buffer_concentration'
        from the run parameters. It performs a lookup via the ingredient controller
        to determine the buffer's pH. If the buffer is not found in the database
        or if the XML parameters are missing, it provides default placeholder
        values (pH 0.0, concentration 0.0) to signal that user completion is required.

        Returns:
            dict: A dictionary containing:
                - 'buffer' (Buffer): A Buffer object initialized with name and pH.
                - 'concentration' (float): The buffer concentration value.
                - 'units' (str): The measurement units (defaults to "mM").
                - 'found' (dict): A mapping of boolean flags indicating if 'name',
                  'conc', and 'units' were successfully parsed from the XML.

        Notes:
            If 'buffer_type' is missing from the XML, a warning is logged and a
            default dictionary with 'found' flags set to False is returned immediately.
        """
        try:
            name = self.get_param("buffer_type", str, required=True)
        except Exception:
            Log.w(
                TAG,
                f"<buffer_type> param could not be found in XML; returning default.",
            )
            return {
                "buffer": Buffer(enc_id=-1, name="None", pH=0.0),
                "concentration": 0.0,
                "units": "mM",
                "found": {"name": False, "conc": False, "units": False},
            }

        buffer_obj = self.ing_ctrl.get_buffer_by_name(name)

        # If unseen in DB, create a placeholder with 0.0 to force user completion
        if buffer_obj is None:
            buffer_ph = 0.0
        else:
            buffer_ph = buffer_obj.pH if buffer_obj.pH is not None else 0.0

        try:
            conc = self.get_param("buffer_concentration", float, required=True)
        except Exception:
            conc = 0.0

        units = self.get_param_attr("buffer_concentration", "units")
        if units is None:
            units = "mM"

        found = {
            "name": name is not None,
            "conc": conc is not None,
            "units": units is not None,
        }

        return {
            "buffer": Buffer(enc_id=-1, name=name, pH=buffer_ph),
            "concentration": conc,
            "units": units,
            "found": found,
        }

    def get_stabilizer(self) -> dict:
        """Constructs and returns stabilizer-related data from the run parameters.

        This method extracts the stabilizer type, concentration, and units from
        the `<params>` section. If specific values are missing, it defaults to
        "None" for the name, 0.0 for the concentration, and "mM" for the units.
        The 'found' dictionary provides metadata on which fields were explicitly
        defined in the input source.

        Returns:
            dict: A dictionary containing:
                - 'stabilizer' (Stabilizer): A Stabilizer object initialized
                  with the found name.
                - 'concentration' (float): The stabilizer concentration value.
                - 'units' (str): The measurement units (defaults to "mM").
                - 'found' (dict): A mapping of boolean flags indicating if
                  'name', 'conc', and 'units' were present in the parameters.
        """
        name = self.get_param("stabilizer_type", str, required=False)
        conc = self.get_param("stabilizer_concentration",
                              float, required=False)
        units = self.get_param_attr(
            "stabilizer_concentration", "units", required=False)
        found = {
            "name": name is not None,
            "conc": conc is not None,
            "units": units is not None,
        }
        if name is None:
            name = "None"
        if conc is None:
            conc = 0.0
        if units is None:
            units = "mM"
        return {
            "stabilizer": Stabilizer(enc_id=-1, name=name),
            "concentration": conc,
            "units": units,
            "found": found,
        }

    def get_surfactant(self) -> dict:
        """Constructs and returns surfactant-related data from the run parameters.

        This method extracts the surfactant type, concentration, and units from
        the `<params>` section. It handles missing data by providing defaults:
        "None" for the name, 0.0 for concentration, and "%w" (weight percent)
        for units. A metadata dictionary 'found' is included to track which
        values were explicitly defined in the source.

        Returns:
            dict: A dictionary containing:
                - 'surfactant' (Surfactant): A Surfactant object initialized
                  with the found name.
                - 'concentration' (float): The surfactant concentration value.
                - 'units' (str): The measurement units (defaults to "%w").
                - 'found' (dict): A mapping of boolean flags indicating if
                  'name', 'conc', and 'units' were present in the parameters.
        """
        name = self.get_param("surfactant_type", str, required=False)
        conc = self.get_param("surfactant_concentration",
                              float, required=False)
        units = self.get_param_attr(
            "surfactant_concentration", "units", required=False)
        found = {
            "name": name is not None,
            "conc": conc is not None,
            "units": units is not None,
        }
        if name is None:
            name = "None"
        if conc is None:
            conc = 0.0
        if units is None:
            units = "%w"
        return {
            "surfactant": Surfactant(enc_id=-1, name=name),
            "concentration": conc,
            "units": units,
            "found": found,
        }

    def get_excipient(self) -> dict:
        """Constructs and returns salt-related data from the run parameters.

        This method extracts the salt type, concentration, and associated units
        from the `<params>` section. If specific values are missing, it applies
        defaults ("None" for name, 0.0 for concentration, and "mM" for units).
        It also tracks which specific fields were successfully located via a
        'found' metadata dictionary.

        Returns:
            dict: A dictionary containing:
                - 'salt' (Salt): A Salt object initialized with the found name.
                - 'concentration' (float): The salt concentration value.
                - 'units' (str): The measurement units for the concentration.
                - 'found' (dict): A mapping of boolean flags indicating if
                  'name', 'conc', and 'units' were explicitly present.
        """
        name = self.get_param("excipient_type", str, required=False)
        conc = self.get_param("excipient_concentration", float, required=False)
        units = self.get_param_attr(
            "excipient_concentration", "units", required=False)
        found = {
            "name": name is not None,
            "conc": conc is not None,
            "units": units is not None,
        }
        if name is None:
            name = "None"
        if conc is None:
            conc = 0.0
        if units is None:
            units = "mM"

        return {
            "excipient": Excipient(enc_id=-1, name=name),
            "concentration": conc,
            "units": units,
            "found": found,
        }

    def get_salt(self) -> dict:
        """Construct and return a `Salt` object with concentration and units from `<params>`."""
        name = self.get_param("salt_type", str, required=False)
        conc = self.get_param("salt_concentration", float, required=False)
        units = self.get_param_attr(
            "salt_concentration", "units", required=False)
        found = {
            "name": name is not None,
            "conc": conc is not None,
            "units": units is not None,
        }
        if name is None:
            name = "None"
        if conc is None:
            conc = 0.0
        if units is None:
            units = "mM"

        return {
            "salt": Salt(enc_id=-1, name=name),
            "concentration": conc,
            "units": units,
            "found": found,
        }

    def get_metrics(self) -> dict[str, str]:
        """Extracts all metric entries from the most recent `<metrics>` section.

        This method searches the XML tree for all occurrences of the `<metrics>`
        tag and selects the last one found. It iterates through the child
        `<metric>` elements, extracting the name, value, and units. If units
        are provided, they are appended to the value string for clarity.

        Returns:
            dict[str, str]: A dictionary where keys are metric names and
                values are the corresponding measurement strings (with units
                included if available). Returns an empty dictionary if no
                `<metrics>` section is found.
        """
        metrics_list = self.root.findall("metrics")
        if not metrics_list:
            return {}

        metrics_elem = metrics_list[-1]
        metrics_dict = {}

        for metric in metrics_elem:
            if metric.tag == "metric":
                name = metric.get("name", "")
                value = metric.get("value", "")
                units = metric.get("units", "")

                if units:
                    metrics_dict[name] = f"{value} {units}"
                else:
                    metrics_dict[name] = value

        return metrics_dict

    def get_signature(self) -> Optional[str]:
        """Retrieves the run signature from the `<metrics>` element attributes.

        This method searches the XML tree for all occurrences of the `<metrics>`
        tag and extracts the 'signature' attribute from the most recent (last)
        instance found. The signature is expected to be a unique identifier
        associated with the run's metric calculation.

        Returns:
            Optional[str]: The signature string if found in the last `<metrics>`
                element; None if no `<metrics>` tags exist or the attribute
                is missing.
        """
        metrics_list = self.root.findall("metrics")
        if not metrics_list:
            return None
        return metrics_list[-1].get("signature")

    def get_audits(self) -> dict[str, tuple[str, str]]:
        """Extracts audit log entries from the most recent `<audits>` XML section.

        This method searches the XML tree for all occurrences of the `<audits>`
        tag and selects only the last one found, assuming it to be the most
        recent record. It iterates through the child `<audit>` elements to
        map specific actions to the user and timestamp associated with them.

        Returns:
            dict[str, tuple[str, str]]: A dictionary where keys are the 'action'
                strings (e.g., "Created", "Modified") and values are 2-tuples
                containing (username, recorded_timestamp). Returns an empty
                dictionary if no `<audits>` section is found.
        """
        audits_list = self.root.findall("audits")
        if not audits_list:
            return {}

        audits_elem = audits_list[-1]  # most recent element
        audits_dict = {}

        for audit in audits_elem:
            if audit.tag == "audit":
                action = audit.get("action", "")
                username = audit.get("username", "")
                recorded = audit.get("recorded", "")

                if action:
                    audits_dict[action] = (username, recorded)

        return audits_dict

    def get_run_notes(self) -> Optional[str]:
        """Retrieves and formats the run notes from the parameters.

        This method looks up the 'notes' key within the parsed parameters. If
        notes are found, it performs a string replacement to convert literal
        backslash-n escape sequences ("\\n") into actual newline characters
        for proper display formatting.

        Returns:
            Optional[str]: The formatted notes string if present; None if the
                parameter is missing or empty.
        """
        notes = self.get_param("notes", str, required=False)
        if notes:
            return notes.replace("\\n", "\n")
        return None

    def get_batch_number(self) -> Optional[str]:
        """Retrieves the batch number from the run parameters.

        This method attempts to look up the 'batch_number' key within the parsed
        parameters. Since this is an optional field, it will not raise an error
        if the key is missing.

        Returns:
            Optional[str]: The batch number as a string if it exists in the
                parameters; None otherwise.
        """
        return self.get_param("batch_number", str, required=False)

    def get_fill_type(self) -> str:
        """Retrieves the fill type configuration from the run parameters.

        This method searches the parsed parameters for a 'fill_type' identifier.
        If the parameter is missing or evaluates to an empty value, it
        defaults to '3', which typically represents a standard fill protocol.

        Returns:
            str: The value of the 'fill_type' parameter if found;
                otherwise returns the default string '3'.
        """
        return self.get_param("fill_type", str, required=False) or "3"

    def is_bioformulation(self) -> bool:
        """Checks if the current run is explicitly flagged as a bioformulation.

        This method queries the parsed parameters for a 'bioformulation' key.
        Since parameters are often stored as strings in the underlying XML/data
        source, it performing a specific check against the string "True".

        Returns:
            bool: True if the 'bioformulation' parameter is present and
                exactly matches the string "True"; False otherwise.
        """
        value = self.get_param("bioformulation", str, required=False)
        return value == "True" if value else False

    def get_viscosity_profile(self) -> ViscosityProfile:
        """Locates and extracts viscosity data from the most recent analysis archive.

        This method searches the instance's base path for zip archives matching the
        pattern 'analyze-[INT].zip' and selects the one with the highest integer index.
        It then reads shear rate, viscosity, and temperature data from the associated
        '*_analyze_out.csv' file within the archive. The extracted viscosities are
        interpolated to match the predefined shear rates in `self.profile_shears`
        before being packaged into a measured `ViscosityProfile`.

        Returns:
            tuple: A tuple containing:
                - ViscosityProfile: The interpolated viscosity profile marked as
                  measured data (units in cP).
                - float: The average temperature calculated from the CSV data.

        Raises:
            FileNotFoundError: If the base path is invalid or missing, if no
                'analyze-*.zip' files are found in the directory, or if the required
                '*_analyze_out.csv' file is missing from the selected archive.
        """
        if not self.base_path or not os.path.isdir(self.base_path):
            raise FileNotFoundError(f"Base path not found: {self.base_path}")

        all_files = os.listdir(self.base_path)
        analyze_zips = [f for f in all_files if re.match(
            r"analyze-\d+\.zip$", f)]

        if not analyze_zips:
            raise FileNotFoundError(
                f"No analyze-*.zip files found in {self.base_path}")
        largest_zip_name = max(
            analyze_zips,
            key=lambda n: int(re.search(r"analyze-(\d+)\.zip", n).group(1)),
        )
        zip_base_name = largest_zip_name[:-4]

        # Get namelist from the analyze zip
        dummy_path = os.path.join(self.base_path, "dummy")
        namelist = SecureOpen.get_namelist(dummy_path, zip_name=zip_base_name)

        csv_files = [n for n in namelist if n.endswith("_analyze_out.csv")]
        if not csv_files:
            raise FileNotFoundError(
                f"No *_analyze_out.csv found inside {largest_zip_name}"
            )

        csv_file_name = csv_files[0]
        csv_path = os.path.join(self.base_path, csv_file_name)
        with SecureOpen(csv_path, "r", zipname=zip_base_name, insecure=True) as csv_f:
            csv_data = np.loadtxt(csv_f, delimiter=",",
                                  skiprows=1, usecols=(0, 2, 4))
            # Handle case where csv has only one row
            if csv_data.ndim == 1:
                csv_data = csv_data.reshape(1, -1)

            shear_rate = csv_data[:, 0]
            viscosity = csv_data[:, 1]
            temperature = csv_data[:, 2]

        shear_rates_list = shear_rate.tolist()
        viscosities_list = viscosity.tolist()
        temp_profile = ViscosityProfile(
            shear_rates=shear_rates_list, viscosities=viscosities_list, units="cP"
        )
        interpolated_viscosities = [
            temp_profile.get_viscosity(sr) for sr in self.profile_shears
        ]
        profile = ViscosityProfile(
            shear_rates=self.profile_shears,
            viscosities=interpolated_viscosities,
            units="cP",
        )

        # Mark as measured data
        profile.is_measured = True

        return profile, np.average(temperature)

    def get_run_name(self) -> Optional[str]:
        """Retrieves the run name from the `<run_info>` XML tag.

        This method attempts to extract the 'name' attribute by first checking if
        the root XML element is `<run_info>`. If it is not, it falls back to
        searching for a `<run_info>` child element. Warnings are logged if the
        tag is completely missing or if it exists but lacks the required 'name'
        attribute.

        Returns:
            Optional[str]: The extracted run name as a string, or None if the
                `<run_info>` tag or its 'name' attribute cannot be found.
        """
        # Check if root element is run_info
        if self.root.tag == "run_info":
            run_name = self.root.get("name")
            if run_name is None:
                Log.w(
                    TAG,
                    "<run_info> root element found but 'name' attribute is missing",
                )
                return None
            return run_name

        # Fallback: search for run_info as a child element
        run_info = self.root.find("run_info")
        if run_info is None:
            Log.w(TAG, "No <run_info> element found in XML")
            return None

        run_name = run_info.get("name")
        if run_name is None:
            Log.w(TAG, "<run_info> element found but 'name' attribute is missing")
            return None

        return run_name

    def get_formulation(self) -> Formulation:
        """Constructs and populates a `Formulation` object using parsed parameters.

        This method aggregates data from various internal parsing helpers to build
        a comprehensive formulation record. It retrieves run identifiers, notes, and
        specific ingredient data (buffer, protein, surfactant, stabilizer, excipient,
        and salt). It also attempts to load viscosity data, defaulting to 25.0°C if
        the profile is missing. Critical missing components (Buffer Type, Protein Type,
        Viscosity Data) are tracked and attached to the resulting object's metadata.

        Returns:
            Formulation: A populated object containing the parsed formulation
                properties, along with a `missing_fields` attribute listing any
                critical data that was not found.
        """
        formulation = Formulation()
        missing_fields = []

        # Set formulation identifiers from Run Info
        run_name = self.get_run_name()
        if run_name:
            formulation.name = run_name

        run_signature = self.get_signature()
        if run_signature:
            formulation.signature = run_signature

        # Get Latest Notes (from the last <params> block)
        notes = self.get_run_notes()
        if notes:
            formulation.notes = notes

        #  Fetch Ingredients & Track Missing Fields

        buffer_data = self.get_buffer()
        if not buffer_data["found"]["name"]:
            missing_fields.append("Buffer Type")

        protein_data = self.get_protein()
        if not protein_data["found"]["name"]:
            missing_fields.append("Protein Type")

        surfactant_data = self.get_surfactant()
        stabilizer_data = self.get_stabilizer()
        excipient_data = self.get_excipient()
        salt_data = self.get_salt()

        # Try to load viscosity profile
        try:
            vp, temp = self.get_viscosity_profile()
            formulation.set_temperature(temp=temp)
            formulation.set_viscosity_profile(profile=vp)
        except (FileNotFoundError, ValueError):
            missing_fields.append("Viscosity Data")
            formulation.set_temperature(25.0)  # Default

        # Populate Formulation
        formulation.set_buffer(
            buffer=buffer_data["buffer"],
            concentration=buffer_data["concentration"],
            units=buffer_data["units"],
        )
        formulation.set_protein(
            protein=protein_data["protein"],
            concentration=protein_data["concentration"],
            units=protein_data["units"],
        )
        formulation.set_surfactant(
            surfactant=surfactant_data["surfactant"],
            concentration=surfactant_data["concentration"],
            units=surfactant_data["units"],
        )
        formulation.set_stabilizer(
            stabilizer=stabilizer_data["stabilizer"],
            concentration=stabilizer_data["concentration"],
            units=stabilizer_data["units"],
        )
        formulation.set_excipient(
            excipient=excipient_data["excipient"],
            concentration=excipient_data["concentration"],
            units=excipient_data["units"],
        )
        formulation.set_salt(
            salt=salt_data["salt"],
            concentration=salt_data["concentration"],
            units=salt_data["units"],
        )

        # Attach missing fields metadata
        formulation.missing_fields = missing_fields

        return formulation
