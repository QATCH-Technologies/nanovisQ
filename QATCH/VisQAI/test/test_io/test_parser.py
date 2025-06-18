"""
test_parser.py

Unit tests for the Parser class, verifying:
    - Initialization behavior when XML file is missing
    - get_text method for successful retrieval, missing tags, and invalid casts
    - get_param method for retrieving required/optional parameters and type casting errors
    - get_param_attr method for retrieving attribute values with required/optional flags
    - Parsing of Protein, Buffer, Surfactant, and Stabilizer parameters into model objects
    - Construction of a Formulation from parsed parameters, viscosity profile, and salt inputs
    - Initialization with a sample XML file if available

Author:
    Paul MacNichol (paul.macnichol@qatchtech.com)

Date:
    2025-06-02

Version:
    1.1
"""

import os
import tempfile
import unittest
import xml.etree.ElementTree as ET

from src.io.parser import Parser
from src.models.ingredient import Protein, Buffer, Stabilizer, Surfactant, Salt
from src.models.formulation import Formulation


class TestParser(unittest.TestCase):
    """Unit tests for the `Parser` class that extracts parameters from XML."""

    SAMPLE_XML_PATH = os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     "test_assets", "sample_xml.xml")
    )

    def setUp(self):
        """Create a temporary XML file containing parameters and initialize a Parser."""
        self.temp_file = tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".xml"
        )
        xml_content = '''<?xml version="1.0"?>
<run_info>
  <params>
    <param name="protein_type" value="BSA" />
    <param name="protein_mw" value="100.0" />
    <param name="protein_pI_mean" value="7.4" />
    <param name="protein_pI_range" value="1.2" />
    <param name="protein_concentration" value="10.0" units="mg/ml" />
    <param name="buffer_type" value="PBS" />
    <param name="buffer_pH" value="7.4" />
    <param name="buffer_concentration" value="20.0" units="M" />
    <param name="surfactant_type" value="Tween20" />
    <param name="surfactant_concentration" value="0.1" units="%" />
    <param name="stabilizer_type" value="None" />
    <param name="stabilizer_concentration" value="0.0" units="M" />
  </params>
</run_info>'''
        self.temp_file.write(xml_content)
        self.temp_file.close()
        self.parser = Parser(self.temp_file.name)

    def tearDown(self):
        """Remove the temporary XML file after each test."""
        try:
            os.remove(self.temp_file.name)
        except OSError:
            pass

    def test_init_file_not_found(self):
        """
        Test that initializing Parser with a non-existent file raises FileNotFoundError.
        """
        with self.assertRaises(FileNotFoundError):
            Parser("nonexistent_file.xml")

    def test_get_text_success(self):
        """
        Test that get_text returns the correct casted value when the tag exists.

        - Create an XML element with a <child> containing "123"
        - Call get_text(elem, 'child', int) and verify it returns integer 123
        """
        elem = ET.fromstring("<parent><child>123</child></parent>")
        result = self.parser.get_text(elem, "child", int)
        self.assertEqual(result, 123)

    def test_get_text_missing_tag(self):
        """
        Test that get_text raises ValueError when the requested tag is missing.

        - Create an XML element with no <child> tag
        - Call get_text(elem, 'child', str) and expect ValueError containing "Missing <child>"
        """
        elem = ET.fromstring("<parent></parent>")
        with self.assertRaises(ValueError) as cm:
            self.parser.get_text(elem, "child", str)
        self.assertIn("Missing <child>", str(cm.exception))

    def test_get_text_invalid_cast(self):
        """
        Test that get_text raises ValueError when the text cannot be cast to the requested type.

        - Create an XML element with <child> containing "abc"
        - Call get_text(elem, 'child', float) and expect ValueError containing "Cannot cast 'abc'"
        """
        elem = ET.fromstring("<parent><child>abc</child></parent>")
        with self.assertRaises(ValueError) as cm:
            self.parser.get_text(elem, "child", float)
        self.assertIn("Cannot cast 'abc'", str(cm.exception))

    def test_get_param_success(self):
        """
        Test that get_param retrieves and casts parameter values correctly.

        - Retrieve 'protein_type' as string, expect "BSA"
        - Retrieve 'protein_mw' as float, expect a float ~100.0
        """
        val = self.parser.get_param("protein_type", str)
        self.assertEqual(val, "BSA")
        num = self.parser.get_param("protein_mw", float)
        self.assertIsInstance(num, float)
        self.assertAlmostEqual(num, 100.0)

    def test_get_param_missing_required(self):
        """
        Test that get_param raises ValueError when a required parameter is missing.

        - Call get_param('nonexistent', str, required=True), expect ValueError
        """
        with self.assertRaises(ValueError):
            self.parser.get_param("nonexistent", str, required=True)

    def test_get_param_missing_not_required(self):
        """
        Test that get_param returns None when an optional parameter is missing.

        - Call get_param('nonexistent', str, required=False), expect None
        """
        val = self.parser.get_param("nonexistent", str, required=False)
        self.assertIsNone(val)

    def test_get_param_cast_error(self):
        """
        Test that get_param raises ValueError when value cannot be cast to requested type.

        - Create a temporary XML with <param name='bad' value='not_a_number' />
        - Call get_param('bad', float), expect ValueError
        """
        root = ET.Element("run_info")
        params = ET.SubElement(root, "params")
        ET.SubElement(params, "param", name="bad", value="not_a_number")
        bad_file = tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".xml"
        )
        bad_file.write(ET.tostring(root, encoding="unicode"))
        bad_file.close()
        bad_parser = Parser(bad_file.name)
        with self.assertRaises(ValueError):
            bad_parser.get_param("bad", float)
        os.remove(bad_file.name)

    def test_get_param_attr_existing(self):
        """
        Test that get_param_attr returns the attribute value when it exists and is required.

        - 'protein_concentration' has units="mg/ml"
        - Call get_param_attr('protein_concentration', 'units', required=True), expect "mg/ml"
        """
        units = self.parser.get_param_attr(
            "protein_concentration", "units", required=True
        )
        self.assertEqual(units, "mg/ml")

    def test_get_param_attr_missing_not_required(self):
        """
        Test that get_param_attr returns None when the attribute is missing but not required.

        - 'protein_type' has no units attribute
        - Call get_param_attr('protein_type', 'units', required=False), expect None
        """
        attr = self.parser.get_param_attr(
            "protein_type", "units", required=False
        )
        self.assertIsNone(attr)

    def test_get_param_attr_missing_required(self):
        """
        Test that get_param_attr raises ValueError when a required attribute is missing.

        - 'protein_type' has no units attribute
        - Call get_param_attr('protein_type', 'units', required=True), expect ValueError
        """
        with self.assertRaises(ValueError):
            self.parser.get_param_attr("protein_type", "units", required=True)

    def test_get_param_attr_no_params(self):
        """
        Test that get_param_attr returns None when the <params> section is absent and not required.

        - Create a temporary XML with no <params>
        - Call get_param_attr('any', 'attr', required=False), expect None
        """
        no_params_file = tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".xml"
        )
        no_params_file.write("<run_info></run_info>")
        no_params_file.close()
        no_parser = Parser(no_params_file.name)
        self.assertIsNone(
            no_parser.get_param_attr("any", "attr", required=False)
        )
        os.remove(no_params_file.name)

    def test_get_protein(self):
        """
        Test that get_protein returns a dict with a Protein instance, concentration, and units.

        - Call get_protein() on the initial parser
        - Verify returned dict has keys: 'protein', 'concentration', 'units'
        - Confirm 'protein' is a Protein with name "BSA", concentration ~10.0, units "mg/ml"
        """
        prot = self.parser.get_protein()
        self.assertIsInstance(prot, dict)
        protein_obj = prot["protein"]
        self.assertIsInstance(protein_obj, Protein)
        self.assertEqual(protein_obj.name, "BSA")
        self.assertAlmostEqual(prot["concentration"], 10.0)
        self.assertEqual(prot["units"], "mg/ml")

    def test_get_buffer(self):
        """
        Test that get_buffer returns a dict with a Buffer instance, concentration, and units.

        - Call get_buffer() on the initial parser
        - Verify returned dict has keys: 'buffer', 'concentration', 'units'
        - Confirm 'buffer' is a Buffer with pH ~7.4, concentration ~20.0, units "M"
        """
        buf = self.parser.get_buffer()
        self.assertIsInstance(buf, dict)
        buffer_obj = buf["buffer"]
        self.assertIsInstance(buffer_obj, Buffer)
        self.assertAlmostEqual(buf["concentration"], 20.0)
        self.assertEqual(buf["units"], "M")

    def test_get_surfactant(self):
        """
        Test that get_surfactant returns a dict with a Surfactant instance, concentration, and units.

        - Call get_surfactant() on the initial parser
        - Verify returned dict has keys: 'surfactant', 'concentration', 'units'
        - Confirm 'surfactant' is a Surfactant with concentration ~0.1, units "%"
        """
        surf = self.parser.get_surfactant()
        self.assertIsInstance(surf, dict)
        surf_obj = surf["surfactant"]
        self.assertIsInstance(surf_obj, Surfactant)
        self.assertAlmostEqual(surf["concentration"], 0.1)
        self.assertEqual(surf["units"], "%")

    def test_get_stabilizer(self):
        """
        Test that get_stabilizer returns a dict with a Stabilizer instance, concentration, and units.

        - Call get_stabilizer() on the initial parser
        - Verify returned dict has keys: 'stabilizer', 'concentration', 'units'
        - Confirm 'stabilizer' is a Stabilizer with concentration ~0.0, units "M"
        """
        stab = self.parser.get_stabilizer()
        self.assertIsInstance(stab, dict)
        stab_obj = stab["stabilizer"]
        self.assertIsInstance(stab_obj, Stabilizer)
        self.assertAlmostEqual(stab["concentration"], 0.0)
        self.assertEqual(stab["units"], "M")

    def test_get_formulation(self):
        """
        Test that get_formulation constructs a Formulation with components, temperature, and viscosity profile.

        - Provide salt (Salt instance), vp dict with 'shear_rate' and 'viscosity', temp float
        - Call get_formulation(vp, temp, salt, salt_concentration, salt_units)
        - Verify returned object is a Formulation
        - Convert to dict and confirm temperature, viscosity_profile.shear_rates/viscosities/units match inputs
        """
        salt = Salt(enc_id=-1, name="nacl")
        vp = {"shear_rate": [100, 1000], "viscosity": [10, 9]}
        temp = 25.0
        form = self.parser.get_formulation(
            vp=vp, temp=temp, salt=salt, salt_concentration=100, salt_units="mg/ml"
        )
        self.assertIsInstance(form, Formulation)
        data = form.to_dict()
        self.assertAlmostEqual(data["temperature"], temp)
        vp_data = data["viscosity_profile"]
        self.assertEqual(vp_data["shear_rates"], vp["shear_rate"])
        self.assertEqual(vp_data["viscosities"], vp["viscosity"])
        self.assertEqual(vp_data["units"], "cP")

    @unittest.skipUnless(os.path.exists(SAMPLE_XML_PATH), "Sample XML not available")
    def test_init_with_sample_xml(self):
        """
        Test that Parser can be initialized with a pre-existing sample XML file.

        - Attempt to load SAMPLE_XML_PATH
        - Verify parser.root is not None
        """
        parser = Parser(self.SAMPLE_XML_PATH)
        self.assertIsNotNone(parser.root)


if __name__ == "__main__":
    unittest.main()
