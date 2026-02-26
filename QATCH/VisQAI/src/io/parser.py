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
    2025-11-05

Version:
    1.6
"""

import os
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any, List, Optional, Type, Union

import numpy as np

try:

    class Log:
        @staticmethod
        def d(tag, msg=""):
            print("DEBUG:", tag, msg)

        @staticmethod
        def i(tag, msg=""):
            print("INFO:", tag, msg)

        @staticmethod
        def w(tag, msg=""):
            print("WARNING:", tag, msg)

        @staticmethod
        def e(tag, msg=""):
            print("ERROR:", tag, msg)

    from src.controller.ingredient_controller import IngredientController
    from src.db.db import Database
    from src.io.file_storage import SecureOpen
    from src.models.formulation import Formulation, ViscosityProfile
    from src.models.ingredient import (
        Buffer,
        Excipient,
        Protein,
        ProteinClass,
        Salt,
        Stabilizer,
        Surfactant,
    )

except (ModuleNotFoundError, ImportError):
    from QATCH.common.logger import Logger as Log
    from QATCH.VisQAI.src.controller.ingredient_controller import IngredientController
    from QATCH.VisQAI.src.db.db import Database
    from QATCH.VisQAI.src.io.file_storage import SecureOpen
    from QATCH.VisQAI.src.models.formulation import Formulation, ViscosityProfile
    from QATCH.VisQAI.src.models.ingredient import (
        Buffer,
        Excipient,
        Protein,
        ProteinClass,
        Salt,
        Stabilizer,
        Surfactant,
    )
TAG = "[Parser]"


class Parser:
    """XML parser for extracting formulation-related parameters and creating model objects.

    The parser expects an XML file with a `<params>` section containing multiple `<param>` elements.
    Each `<param>` must have a `name` attribute and may have a `value` attribute or other attributes
    (e.g., `units`). This class provides methods to retrieve values and build `Protein`, `Buffer`,
    `Stabilizer`, `Surfactant`, and `Formulation` instances based on the XML content.
    """

    def __init__(self, xml_path: str = None):
        """Initialize the parser resources.

        Args:
            xml_path (str, optional): Legacy argument for backward compatibility.
                                      If provided, it initializes state immediately,
                                      but usage of parse() is preferred.
        """
        self.database = Database(parse_file_key=True)
        self.ing_ctrl = IngredientController(self.database)
        self.profile_shears = [1e2, 1e3, 1e4, 1e5, 15000000]

        # State variables for the current file being parsed
        self.xml_path = None
        self.base_path = None
        self.root = None
        self.params = None

        # Backward compatibility for legacy code that instantiates with a path
        if xml_path:
            self._load_state(xml_path)

    def parse(self, source: Union[str, List[str]]) -> List[Formulation]:
        """
        Parse one or more directories/files to extract Formulation objects.

        Args:
            source (Union[str, List[str]]): A single path string or a list of path strings.
                                            Can be directories (searched recursively) or specific XML files.

        Returns:
            List[Formulation]: A list of valid Formulation objects extracted from the source(s).
        """
        if isinstance(source, str):
            sources = [source]
        else:
            sources = source

        formulations = []

        # Collect potential XML files
        xml_candidates = []
        for path_str in sources:
            path = Path(path_str)
            if not path.exists():
                Log.w(TAG, f"Source path does not exist: {path}")
                continue

            if path.is_file() and path.suffix.lower() == ".xml":
                xml_candidates.append(path)
            elif path.is_dir():
                # Walk directory to find XML files
                # We assume any XML file might be a run file
                for root, _, files in os.walk(path):
                    for file in files:
                        if file.lower().endswith(".xml"):
                            xml_candidates.append(Path(root) / file)

        # Process each candidate
        for xml_file in xml_candidates:
            try:
                self._load_state(str(xml_file))

                # --- CHECK: Only parse bioformulations ---
                if not self.is_bioformulation():
                    Log.i(TAG, f"Skipping non-bioformulation file: {xml_file.name}")
                    continue
                # -----------------------------------------

                # If load_state succeeds and it IS a bioformulation
                # Extract the formulation
                form = self.get_formulation()
                if form:
                    formulations.append(form)
            except Exception as e:
                # Log error but continue parsing other files
                Log.w(TAG, f"Skipping file {xml_file.name}: {e}")
                continue

        return formulations

    def _load_state(self, xml_path_str: str):
        """
        Internal method to load the XML and setup state variables (root, params) for a specific file.

        Raises:
            FileNotFoundError: If XML or required sibling files/folders are missing.
            ET.ParseError: If XML is malformed.
        """
        input_path = Path(xml_path_str)

        # Logic adapted from original __init__:
        # The parser seems to rely on the XML file residing in a folder that also contains data (capture.zip or analyze zips)
        self.base_path = input_path.parent
        self.xml_path = input_path

        if not self.xml_path.exists():
            raise FileNotFoundError(f"XML not found at path `{self.xml_path}`.")

        # Note: Original code checked for capture.zip at the input path if input was expected to be zip.
        # Here we assume we are finding XMLs directly.
        # However, get_viscosity_profile() requires valid base_path with analyze zips.

        tree = ET.parse(self.xml_path)
        self.root = tree.getroot()

        params_list = self.root.findall("params")
        if not params_list:
            self.params = None
        else:
            # Use the last <params> section if multiple are present
            self.params = params_list[-1]

    def get_text(self, elem: ET.Element, tag: str, cast_type: Type) -> Any:
        """Retrieve and cast the text content of a child element."""
        txt = elem.findtext(tag)
        if txt is None:
            raise ValueError(f"Missing <{tag}> in <{elem.tag}>")
        try:
            return cast_type(txt)
        except ValueError:
            raise ValueError(f"Cannot cast '{txt}' of <{tag}> to {cast_type}")

    def get_param(self, name: str, cast_type: Type = str, required: bool = True) -> Any:
        """Retrieve a parameter value by name from the `<params>` section and cast it."""
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
            raise ValueError(f"Cannot cast param '{name}' value '{val}' to {cast_type}")

    def get_param_attr(
        self, name: str, attr: str, required: bool = False
    ) -> Optional[str]:
        """Retrieve a specific attribute from a `<param>` element."""
        if self.params is None:
            return None
        el = self.params.find(f"param[@name='{name}']")
        if el is None or attr not in el.attrib:
            if required:
                raise ValueError(f"Param '{name}' missing attribute '{attr}'")
            return None
        return el.get(attr)

    def get_protein(self) -> dict:
        """Construct and return a `Protein` object with concentration and units from `<params>`."""
        try:
            name = self.get_param("protein_type", str, required=True)
        except Exception:
            Log.w(
                TAG,
                f"<protein_type> param could not be found in run XML; returning default.",
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

        # If unseen in DB, create a placeholder with 0s to force user completion
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

        units = self.get_param_attr("protein_concentration", "units", required=False)
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
        """Construct and return a `Buffer` object with concentration and units from `<params>`."""
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
        """Construct and return a `Stabilizer` object with concentration and units from `<params>`."""
        name = self.get_param("stabilizer_type", str, required=False)
        conc = self.get_param("stabilizer_concentration", float, required=False)
        units = self.get_param_attr("stabilizer_concentration", "units", required=False)
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
        """Construct and return a `Surfactant` object with concentration and units from `<params>`."""
        name = self.get_param("surfactant_type", str, required=False)
        conc = self.get_param("surfactant_concentration", float, required=False)
        units = self.get_param_attr("surfactant_concentration", "units", required=False)
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
        """Construct and return a `Excipient` object with concentration and units from `<params>`."""
        name = self.get_param("excipient_type", str, required=False)
        conc = self.get_param("excipient_concentration", float, required=False)
        units = self.get_param_attr("excipient_concentration", "units", required=False)
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
        units = self.get_param_attr("salt_concentration", "units", required=False)
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
        """Extract all metrics from the most recent <metrics> section."""
        metrics_list = self.root.findall("metrics")
        if not metrics_list:
            return {}

        metrics_elem = metrics_list[-1]  # most recent element
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
        """Retrieve the signature from the <metrics> element attributes."""
        # Signature is typically an attribute of the <metrics> tag itself
        # e.g., <metrics signature="12345...">
        metrics_list = self.root.findall("metrics")
        if not metrics_list:
            return None

        # Use the last metrics block found
        return metrics_list[-1].get("signature")

    def get_audits(self) -> dict[str, tuple[str, str]]:
        """Extract all audit entries from the most recent <audits> section."""
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
        """Retrieve the run notes from params."""
        notes = self.get_param("notes", str, required=False)
        if notes:
            return notes.replace("\\n", "\n")
        return None

    def get_batch_number(self) -> Optional[str]:
        """Retrieve the batch number from params."""
        return self.get_param("batch_number", str, required=False)

    def get_fill_type(self) -> str:
        """Retrieve the fill type from params, defaulting to '3'."""
        return self.get_param("fill_type", str, required=False) or "3"

    def is_bioformulation(self) -> bool:
        """Check if this run is marked as a bioformulation."""
        value = self.get_param("bioformulation", str, required=False)
        return value == "True" if value else False

    def get_viscosity_profile(self) -> ViscosityProfile:
        """
        Locate and extract viscosity data from the largest analyze-[INT].zip file.
        """
        if not self.base_path or not os.path.isdir(self.base_path):
            raise FileNotFoundError(f"Base path not found: {self.base_path}")

        all_files = os.listdir(self.base_path)
        analyze_zips = [f for f in all_files if re.match(r"analyze-\d+\.zip$", f)]

        if not analyze_zips:
            raise FileNotFoundError(f"No analyze-*.zip files found in {self.base_path}")
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
            csv_data = np.loadtxt(csv_f, delimiter=",", skiprows=1, usecols=(0, 2, 4))
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
        """Retrieve the run name from the <run_info> tag."""
        # Check if root element is run_info
        if self.root.tag == "run_info":
            run_name = self.root.get("name")
            if run_name is None:
                Log.w(
                    TAG, "<run_info> root element found but 'name' attribute is missing"
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
        """Construct a `Formulation` object using parsed parameters."""
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

        # --- Fetch Ingredients & Track Missing Fields ---

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
        except FileNotFoundError:
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
