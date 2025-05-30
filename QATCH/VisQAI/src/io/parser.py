import os
import xml.etree.ElementTree as ET
from typing import Any, Optional, Type
try:
    from src.models.ingredient import (
        Protein, Buffer, Stabilizer, Surfactant, Salt
    )
    from src.models.formulation import ViscosityProfile, Formulation
except (ModuleNotFoundError, ImportError):
    from QATCH.VisQAI.src.models.ingredient import (
        Protein, Buffer, Stabilizer, Surfactant, Salt
    )
    from QATCH.VisQAI.src.models.formulation import ViscosityProfile, Formulation
TAG = "[Parser]"


class Parser:
    def __init__(self, xml_path: str):
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f'XML not found at path `{xml_path}`.')
        tree = ET.parse(xml_path)
        self.root = tree.getroot()

        params_list = self.root.findall('params')
        if not params_list:
            self.params = None
        else:
            self.params = params_list[-1]

    def get_text(self, elem: ET.Element, tag: str, cast_type: Type) -> Any:
        txt = elem.findtext(tag)
        if txt is None:
            raise ValueError(f"Missing <{tag}> in <{elem.tag}>")
        try:
            return cast_type(txt)
        except ValueError:
            raise ValueError(f"Cannot cast '{txt}' of <{tag}> to {cast_type}")

    def get_param(self,
                  name: str,
                  cast_type: Type = str,
                  required: bool = True
                  ) -> Any:
        if self.params is None:
            if required:
                raise ValueError("No <params> section available")
            return None

        el = self.params.find(f"param[@name='{name}']")
        if el is None:
            if required:
                raise ValueError(f"Missing param name='{name}' in <params>")
            return None

        val = el.get('value')
        if val is None:
            raise ValueError(f"Param '{name}' has no value attribute")
        try:
            return cast_type(val)
        except ValueError:
            raise ValueError(
                f"Cannot cast param '{name}' value '{val}' to {cast_type}")

    def get_param_attr(self,
                       name: str,
                       attr: str,
                       required: bool = False
                       ) -> Optional[str]:
        if self.params is None:
            return None
        el = self.params.find(f"param[@name='{name}']")
        if el is None or attr not in el.attrib:
            if required:
                raise ValueError(f"Param '{name}' missing attribute '{attr}'")
            return None
        return el.get(attr)

    def get_protein(self) -> Protein:
        name = self.get_param('protein_type', str)
        molecular_weight = self.get_param('protein_mw', float)
        pI_mean = self.get_param("protein_pI_mean", float)
        pI_range = self.get_param("protein_pI_range", float)
        conc = self.get_param('protein_concentration', float)
        units = self.get_param_attr(
            'protein_concentration', 'units', required=True)
        return {
            "protein": Protein(enc_id=-1,
                               name=name,
                               molecular_weight=molecular_weight,
                               pI_mean=pI_mean,
                               pI_range=pI_range),
            "concentration": conc,
            "units": units
        }

    def get_buffer(self) -> Buffer:
        name = self.get_param('buffer_type', str)
        pH = self.get_param("buffer_pH", float)
        conc = self.get_param('buffer_concentration', float)
        units = self.get_param_attr('buffer_concentration', 'units')
        return {
            "buffer": Buffer(enc_id=-1,
                             name=name,
                             pH=pH),
            "concentration": conc,
            "units": units,
        }

    def get_stabilizer(self) -> Stabilizer:

        name = self.get_param('stabilizer_type', str)
        conc = self.get_param('stabilizer_concentration', float)
        units = self.get_param_attr('stabilizer_concentration', 'units')
        return {"stabilizer": Stabilizer(enc_id=-1, name=name),
                "concentration": conc,
                "units": units}

    def get_surfactant(self) -> Surfactant:

        name = self.get_param('surfactant_type', str)
        conc = self.get_param('surfactant_concentration', float)
        units = self.get_param_attr('surfactant_concentration', 'units')
        return {"surfactant": Surfactant(enc_id=-1, name=name),
                "concentration": conc,
                "units": units}

    def get_formulation(self, vp: dict, temp: float, salt: Salt, salt_concentration: float, salt_units) -> Formulation:
        formulation = Formulation()
        buffer = self.get_buffer()
        protein = self.get_protein()
        surfactant = self.get_surfactant()
        stabilizer = self.get_stabilizer()
        formulation.set_buffer(
            buffer=buffer['buffer'], concentration=buffer['concentration'], units=buffer['units'])
        formulation.set_protein(
            protein=protein['protein'], concentration=protein['concentration'], units=protein['units'])
        formulation.set_surfactant(
            surfactant=surfactant['surfactant'], concentration=surfactant['concentration'], units=surfactant['units'])
        formulation.set_stabilizer(
            stabilizer=stabilizer['stabilizer'], concentration=stabilizer['concentration'], units=stabilizer['units'])
        formulation.set_temperature(temp=temp)
        formulation.set_salt(
            salt=salt, concentration=salt_concentration, units=salt_units)
        viscosity_profile = ViscosityProfile(
            shear_rates=vp['shear_rate'], viscosities=vp['viscosity'], units='cp')
        formulation.set_viscosity_profile(profile=viscosity_profile)
        return formulation
