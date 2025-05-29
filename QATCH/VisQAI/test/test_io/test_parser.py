import os
import tempfile
import unittest
import xml.etree.ElementTree as ET

from src.io.parser import Parser
from src.models.ingredient import Protein, Buffer, Stabilizer, Surfactant, Salt
from src.models.formulation import Formulation


class TestParser(unittest.TestCase):
    SAMPLE_XML_PATH = os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     'test_assets', 'sample_xml.xml')
    )

    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(
            mode='w', delete=False, suffix='.xml')
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
        try:
            os.remove(self.temp_file.name)
        except OSError:
            pass

    def test_init_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            Parser('nonexistent_file.xml')

    def test_get_text_success(self):
        elem = ET.fromstring('<parent><child>123</child></parent>')
        result = self.parser.get_text(elem, 'child', int)
        self.assertEqual(result, 123)

    def test_get_text_missing_tag(self):
        elem = ET.fromstring('<parent></parent>')
        with self.assertRaises(ValueError) as cm:
            self.parser.get_text(elem, 'child', str)
        self.assertIn("Missing <child>", str(cm.exception))

    def test_get_text_invalid_cast(self):
        elem = ET.fromstring('<parent><child>abc</child></parent>')
        with self.assertRaises(ValueError) as cm:
            self.parser.get_text(elem, 'child', float)
        self.assertIn("Cannot cast 'abc'", str(cm.exception))

    def test_get_param_success(self):
        val = self.parser.get_param('protein_type', str)
        self.assertEqual(val, 'BSA')
        num = self.parser.get_param('protein_mw', float)
        self.assertIsInstance(num, float)
        self.assertAlmostEqual(num, 100.0)

    def test_get_param_missing_required(self):
        with self.assertRaises(ValueError):
            self.parser.get_param('nonexistent', str, required=True)

    def test_get_param_missing_not_required(self):
        val = self.parser.get_param('nonexistent', str, required=False)
        self.assertIsNone(val)

    def test_get_param_cast_error(self):
        root = ET.Element('run_info')
        params = ET.SubElement(root, 'params')
        ET.SubElement(params, 'param', name='bad', value='not_a_number')
        bad_file = tempfile.NamedTemporaryFile(
            mode='w', delete=False, suffix='.xml')
        bad_file.write(ET.tostring(root, encoding='unicode'))
        bad_file.close()
        bad_parser = Parser(bad_file.name)
        with self.assertRaises(ValueError):
            bad_parser.get_param('bad', float)
        os.remove(bad_file.name)

    def test_get_param_attr_existing(self):
        units = self.parser.get_param_attr(
            'protein_concentration', 'units', required=True)
        self.assertEqual(units, 'mg/ml')

    def test_get_param_attr_missing_not_required(self):
        attr = self.parser.get_param_attr(
            'protein_type', 'units', required=False)
        self.assertIsNone(attr)

    def test_get_param_attr_missing_required(self):
        with self.assertRaises(ValueError):
            self.parser.get_param_attr('protein_type', 'units', required=True)

    def test_get_param_attr_no_params(self):
        no_params_file = tempfile.NamedTemporaryFile(
            mode='w', delete=False, suffix='.xml')
        no_params_file.write('<run_info></run_info>')
        no_params_file.close()
        no_parser = Parser(no_params_file.name)
        self.assertIsNone(no_parser.get_param_attr(
            'any', 'attr', required=False))
        os.remove(no_params_file.name)

    def test_get_protein(self):
        prot = self.parser.get_protein()
        self.assertIsInstance(prot, dict)
        protein_obj = prot['protein']
        self.assertIsInstance(protein_obj, Protein)
        self.assertEqual(protein_obj.name, 'BSA')
        self.assertAlmostEqual(prot['concentration'], 10.0)
        self.assertEqual(prot['units'], 'mg/ml')

    def test_get_buffer(self):
        buf = self.parser.get_buffer()
        self.assertIsInstance(buf, dict)
        buffer_obj = buf['buffer']
        self.assertIsInstance(buffer_obj, Buffer)
        self.assertAlmostEqual(buf['concentration'], 20.0)
        self.assertEqual(buf['units'], 'M')

    def test_get_surfactant(self):
        surf = self.parser.get_surfactant()
        self.assertIsInstance(surf, dict)
        surf_obj = surf['surfactant']
        self.assertIsInstance(surf_obj, Surfactant)
        self.assertAlmostEqual(surf['concentration'], 0.1)
        self.assertEqual(surf['units'], '%')

    def test_get_stabilizer(self):
        stab = self.parser.get_stabilizer()
        self.assertIsInstance(stab, dict)
        stab_obj = stab['stabilizer']
        self.assertIsInstance(stab_obj, Stabilizer)
        self.assertAlmostEqual(stab['concentration'], 0.0)
        self.assertEqual(stab['units'], 'M')

    def test_get_formulation(self):
        salt = Salt(enc_id=-1, name='nacl')
        vp = {'shear_rate': [100, 1000], 'viscosity': [10, 9]}
        temp = 25.0
        form = self.parser.get_formulation(
            vp=vp, temp=temp, salt=salt, salt_concentration=100, salt_units='mg/ml')
        self.assertIsInstance(form, Formulation)
        data = form.to_dict()
        self.assertAlmostEqual(data['temperature'], temp)
        vp_data = data['viscosity_profile']
        self.assertEqual(vp_data['shear_rates'], vp['shear_rate'])
        self.assertEqual(vp_data['viscosities'], vp['viscosity'])
        self.assertEqual(vp_data['units'], 'cp')

    @unittest.skipUnless(os.path.exists(SAMPLE_XML_PATH), "Sample XML not available")
    def test_init_with_sample_xml(self):
        parser = Parser(self.SAMPLE_XML_PATH)
        self.assertIsNotNone(parser.root)


if __name__ == '__main__':
    unittest.main()
