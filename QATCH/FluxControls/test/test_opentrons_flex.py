import os
import unittest
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.opentrons_flex import OpentronsFlex
from src.pipette import Pipette
from src.commands import Commands
from src.runs import Runs
from src.labware import Labware
from src.standard_labware import StandardLabware


def dummy_response(data):
    return {"data": data}


class TestOpentronsFlex(unittest.TestCase):
    def setUp(self):
        # Valid MAC for testing
        self.mac = "AA:BB:CC:DD:EE:FF"
        # Patch subprocess.run to always succeed for ping/arp commands
        self.patcher_run = patch("src.opentrons_flex.subprocess.run")
        self.mock_run = self.patcher_run.start()
        self.mock_run.return_value = MagicMock(
            returncode=0, stdout="dummy output", stderr=""
        )
        # Patch find_ip to avoid real ARP parsing during init
        self.patcher_find_ip = patch.object(
            OpentronsFlex, "find_ip", return_value="127.0.0.1"
        )
        self.mock_find_ip = self.patcher_find_ip.start()
        # Patch update_available_protocols to noop during init
        self.patcher_update = patch.object(
            OpentronsFlex, "update_available_protocols", return_value={}
        )
        self.mock_update = self.patcher_update.start()
        # Instantiate
        self.flex = OpentronsFlex(mac_address=self.mac)
        # Stop patch for update_available_protocols so original can be tested later
        self.patcher_update.stop()

    def tearDown(self):
        self.patcher_run.stop()
        self.patcher_find_ip.stop()
        # ensure update patch is stopped
        try:
            self.patcher_update.stop()
        except RuntimeError:
            pass

    def test_set_robot_mac_address_invalid(self):
        # it won't blow up with TypeError.
        with self.assertRaises(ValueError):
            OpentronsFlex(mac_address="invalid_mac")

    def test_set_robot_ipv4_success(self):
        ip = "192.168.1.100"
        # ping returns 0
        self.flex._set_robot_ipv4(ip)
        self.assertEqual(self.flex._get_robot_ipv4(), ip)

    def test_set_robot_ipv4_failure(self):
        # simulate ping failure
        self.mock_run.return_value = MagicMock(returncode=1)
        with self.assertRaises(ConnectionError):
            self.flex._set_robot_ipv4("192.168.1.101")

    def test_find_ip_parses_arp(self):
        # Test find_ip logic with real method
        raw = object.__new__(OpentronsFlex)
        raw._robot_mac_address = self.mac.lower()
        arp_output = (
            "Interface: 192.168.0.1 --- 0x6\n"
            "  Internet Address      Physical Address      Type\n"
            "  192.168.0.50         aa-bb-cc-dd-ee-ff     dynamic\n"
        )
        self.mock_run.return_value = MagicMock(returncode=0, stdout=arp_output)
        ip = raw.find_ip()
        self.assertEqual(ip, "192.168.0.50")

    def test_find_ip_not_found(self):
        raw = object.__new__(OpentronsFlex)
        raw._robot_mac_address = self.mac.lower()
        self.mock_run.return_value = MagicMock(returncode=0, stdout="no match here")
        with self.assertRaises(Exception):
            raw.find_ip()

    @patch.object(Commands, "load_pipette")
    @patch.object(Commands, "send_command")
    def test_load_pipette_sets_id(self, mock_send, mock_load):
        # Prepare send_command response
        mock_send.return_value = {"data": {"result": {"pipetteId": "pipette-123"}}}
        # Load to left
        self.flex.load_pipette(Pipettes.P10_SINGLE, Pipettes.LEFT_MOUNT)
        left = self.flex._get_left_pipette()
        self.assertEqual(left.id, "pipette-123")
        mock_load.assert_called_once()

    @patch.object(Labware, "__init__", return_value=None)
    @patch.object(Commands, "load_labware")
    @patch.object(Commands, "send_command")
    def test_load_labware_success(self, mock_send, mock_load_labware, mock_lab_init):
        # prepare labware attributes
        labware_inst = Labware.__new__(Labware)
        labware_inst.display_name = "TestLab"
        labware_inst.load_name = "test_lab"
        labware_inst.name_space = "test_ns"
        labware_inst.version = 1
        labware_inst.location = DeckLocations.A1
        # ensure no labware present
        self.flex.available_labware[DeckLocations.A1] = None
        mock_send.return_value = {"data": {"result": {"labwareId": "labware-456"}}}
        self.flex.load_labware(DeckLocations.A1, "definition")
        self.assertEqual(
            self.flex.available_labware[DeckLocations.A1].id, "labware-456"
        )
        mock_load_labware.assert_called_once()

    def test_pickup_tip_non_tiprack(self):
        fake_labware = MagicMock(
            is_tiprack=False, display_name="Fake", get_location=lambda: DeckLocations.A1
        )
        fake_pipette = MagicMock(get_mount_position=lambda: MountPositions.LEFT_MOUNT)
        with self.assertRaises(Exception):
            self.flex.pickup_tip(fake_labware, fake_pipette)

    @patch.object(Commands, "aspirate")
    @patch.object(Commands, "send_command")
    def test_aspirate_calls_command(self, mock_send, mock_aspire):
        mock_send.return_value = "aspire-response"
        fake_labware = MagicMock(
            get_display_name=lambda: "L", get_location=lambda: DeckLocations.A1
        )
        fake_pipette = MagicMock(
            get_id=lambda: "p1", get_mount_position=lambda: MountPositions.LEFT_MOUNT
        )
        with patch.object(self.flex, "validate_configuration"):
            resp = self.flex.aspirate(
                fake_labware, fake_pipette, flow_rate=5, volume=10
            )
        self.assertEqual(resp, "aspire-response")
        mock_aspire.assert_called_once()

    @patch.object(Commands, "dispense")
    @patch.object(Commands, "send_command")
    def test_dispense_calls_command(self, mock_send, mock_dispense):
        mock_send.return_value = "dispense-response"
        fake_labware = MagicMock(
            get_display_name=lambda: "L", get_location=lambda: DeckLocations.A1
        )
        fake_pipette = MagicMock(
            get_id=lambda: "p1", get_mount_position=lambda: MountPositions.LEFT_MOUNT
        )
        with patch.object(self.flex, "validate_configuration"):
            resp = self.flex.dispense(
                fake_labware, fake_pipette, flow_rate=5, volume=10
            )
        self.assertEqual(resp, "dispense-response")
        mock_dispense.assert_called_once()

    @patch.object(Commands, "blowout")
    @patch.object(Commands, "send_command")
    def test_blowout_calls_command(self, mock_send, mock_blowout):
        mock_send.return_value = "blowout-response"
        fake_labware = MagicMock(
            get_display_name=lambda: "L", get_location=lambda: DeckLocations.A1
        )
        fake_pipette = MagicMock(
            get_id=lambda: "p1", get_mount_position=lambda: MountPositions.LEFT_MOUNT
        )
        with patch.object(self.flex, "validate_configuration"):
            resp = self.flex.blowout(fake_labware, fake_pipette, flow_rate=5)
        self.assertEqual(resp, "blowout-response")
        mock_blowout.assert_called_once()

    @patch.object(Commands, "drop_tip")
    @patch.object(Commands, "send_command")
    def test_drop_tip_calls_command(self, mock_send, mock_drop):
        mock_send.return_value = "drop-response"
        fake_labware = MagicMock(
            get_display_name=lambda: "L", get_location=lambda: DeckLocations.A1
        )
        fake_pipette = MagicMock(get_id=lambda: "p1")
        with patch.object(self.flex, "validate_configuration"):
            resp = self.flex.drop_tip(fake_labware, fake_pipette)
        self.assertEqual(resp, "drop-response")
        mock_drop.assert_called_once()

    @patch.object(Commands, "move_to_coordinates")
    @patch.object(Commands, "send_command")
    def test_move_to_coordinates(self, mock_send, mock_move):
        mock_send.return_value = "move-response"
        fake_pipette = MagicMock(
            get_id=lambda: "p1", get_mount_position=lambda: MountPositions.LEFT_MOUNT
        )
        with patch.object(self.flex, "validate_configuration"):
            resp = self.flex.move_to_coordiantes(
                fake_pipette, x=1, y=2, z=3, min_z_height=0.5, force_direct=True
            )
        self.assertEqual(resp, "move-response")
        mock_move.assert_called_once()

    @patch.object(Commands, "move_to_well")
    @patch.object(Commands, "send_command")
    def test_move_to_well(self, mock_send, mock_move):
        mock_send.return_value = "well-response"
        fake_labware = MagicMock(display_name="L", location=DeckLocations.A1)
        fake_pipette = MagicMock(get_id=lambda: "p1")
        with patch.object(self.flex, "validate_configuration"):
            resp = self.flex.move_to_well(fake_labware, fake_pipette)
        self.assertEqual(resp, "well-response")
        mock_move.assert_called_once()

    @patch.object(Commands, "move_relative")
    @patch.object(Commands, "send_command")
    def test_move_relative(self, mock_send, mock_move):
        mock_send.return_value = "rel-response"
        fake_pipette = MagicMock(get_id=lambda: "p1")
        resp = self.flex.move_relative(fake_pipette, distance=5, axis=None)
        self.assertEqual(resp, "rel-response")
        mock_move.assert_called_once()

    def test_run_protocol_invalid(self):
        self.flex.available_protocols = {}
        with self.assertRaises(ValueError):
            self.flex.run_protocol("nonexistent")

    @patch.object(Runs, "run_protocol")
    @patch.object(OpentronsFlex, "play_run")
    def test_run_protocol_success(self, mock_play, mock_runprot):
        self.flex.available_protocols = {"prot": {"id": "p1"}}
        mock_runprot.return_value = "run1"
        mock_play.return_value = dummy_response({"id": "run1"})
        resp = self.flex.run_protocol("prot")
        self.assertEqual(resp, "run1")

    @patch.object(Runs, "delete_protocol")
    def test_delete_protocol_invalid(self, mock_delete):
        self.flex.available_protocols = {}
        with self.assertRaises(ValueError):
            self.flex.delete_protocol("x")

    @patch.object(Runs, "delete_protocol")
    @patch.object(OpentronsFlex, "update_available_protocols")
    def test_delete_protocol_success(self, mock_update, mock_delete):
        self.flex.available_protocols = {"p": {"id": "id1"}}
        mock_delete.return_value = "deleted"
        resp = self.flex.delete_protocol("p")
        self.assertEqual(resp, "deleted")
        mock_delete.assert_called_once()

    def test_upload_protocol_missing(self):
        with self.assertRaises(Exception):
            self.flex.upload_protocol("nonexistent_file.py")

    @patch.object(Runs, "upload_protocol")
    @patch.object(OpentronsFlex, "update_available_protocols")
    def test_upload_protocol_success(self, mock_update, mock_upload):
        mock_upload.return_value = "upl"
        # create dummy file
        path = os.path.abspath(__file__)
        resp = self.flex.upload_protocol(path)
        self.assertEqual(resp, "upl")

    def test_upload_protocol_custom_labware_missing_primary(self):
        with self.assertRaises(Exception):
            self.flex.upload_protocol_custom_labware("no.py", "a.json")

    def test_upload_protocol_custom_labware_missing_labware(self):
        # create temp protocol file
        proto = os.path.abspath(__file__)
        with self.assertRaises(Exception):
            self.flex.upload_protocol_custom_labware(proto, "no.json")

    @patch.object(Runs, "upload_protocol_custom_labware")
    @patch.object(OpentronsFlex, "update_available_protocols")
    def test_upload_protocol_custom_labware_success(self, mock_update, mock_upload):
        proto = os.path.abspath(__file__)
        lab = os.path.abspath(__file__)
        mock_upload.return_value = "upl-c"
        resp = self.flex.upload_protocol_custom_labware(proto, lab)
        self.assertEqual(resp, "upl-c")

    @patch.object(Runs, "get_protocols_list")
    def test_get_protocol_list(self, mock_list):
        mock_list.return_value = ["a", "b"]
        resp = self.flex.get_protocol_list()
        self.assertEqual(resp, ["a", "b"])

    def test_update_available_protocols(self):
        # bypass init stub for this test
        flex = OpentronsFlex.__new__(OpentronsFlex)
        # craft protocol list
        now = datetime.utcnow()
        older = now.replace(year=now.year - 1).isoformat() + "Z"
        newer = now.isoformat() + "Z"
        flex.get_protocol_list = MagicMock(
            return_value=[
                {"metadata": {"protocolName": "x"}, "id": "1", "createdAt": older},
                {"metadata": {"protocolName": "x"}, "id": "2", "createdAt": newer},
                {"metadata": {"protocolName": "y"}, "id": "3", "createdAt": older},
            ]
        )
        result = OpentronsFlex.update_available_protocols(flex)
        self.assertEqual(result, {"x": "2", "y": "3"})

    @patch.object(Runs, "delete_run")
    def test_delete_run(self, mock_delete):
        mock_delete.return_value = "dr"
        resp = self.flex.delete_run(5)
        self.assertEqual(resp, "dr")

    @patch.object(Runs, "get_run_status")
    def test_get_run_status(self, mock_status):
        mock_status.return_value = "ok"
        resp = self.flex.get_run_status("r1")
        self.assertEqual(resp, "ok")

    @patch.object(Runs, "get_runs_list")
    def test_get_run_list(self, mock_list):
        mock_list.return_value = ["run1"]
        resp = self.flex.get_run_list()
        self.assertEqual(resp, ["run1"])

    @patch.object(Runs, "pause_run")
    def test_pause_run(self, mock_pause):
        mock_pause.return_value = "p"
        resp = self.flex.pause_run(1)
        self.assertEqual(resp, "p")

    @patch.object(Runs, "play_run")
    def test_play_run(self, mock_play):
        mock_play.return_value = "pl"
        resp = self.flex.play_run(1)
        self.assertEqual(resp, "pl")

    @patch.object(Runs, "stop_run")
    def test_stop_run(self, mock_stop):
        mock_stop.return_value = "st"
        resp = self.flex.stop_run("1")
        self.assertEqual(resp, "st")

    @patch.object(Runs, "resume_run")
    def test_resume_run(self, mock_resume):
        mock_resume.return_value = "rs"
        resp = self.flex.resume_run("1")
        self.assertEqual(resp, "rs")

    @patch.object(Runs, "set_lights")
    def test_lights_on_off(self, mock_set):
        mock_set.return_value = "ok"
        on = self.flex.lights_on()
        off = self.flex.lights_off()
        self.assertEqual(on, "ok")
        self.assertEqual(off, "ok")

    def test_flash_lights_invalid(self):
        with self.assertRaises(ValueError):
            self.flex.flash_lights(0)

    @patch.object(OpentronsFlex, "lights_on")
    @patch.object(OpentronsFlex, "lights_off")
    def test_flash_lights_success(self, mock_off, mock_on):
        self.flex.flash_lights(2)
        self.assertEqual(mock_on.call_count, 2)
        self.assertEqual(mock_off.call_count, 2)

    @patch.object(Runs, "get_lights")
    def test_lights_status(self, mock_get):
        mock_get.return_value = "stat"
        resp = self.flex.lights_status()
        self.assertEqual(resp, "stat")

    @patch.object(Runs, "create_run_from_protocol")
    def test_create_run(self, mock_create):
        mock_create.return_value = {"data": {"id": "newR"}}
        rid = self.flex.create_run()
        self.assertEqual(rid, "newR")

    @patch.object(Runs, "home")
    def test_home(self, mock_home):
        mock_home.return_value = "hm"
        resp = self.flex.home()
        self.assertEqual(resp, "hm")

    def test_validate_configuration(self):
        # set a valid labware and pipette
        lw = MagicMock(
            get_location=lambda: DeckLocations.A1, get_display_name=lambda: "L"
        )
        pip = MagicMock(
            get_mount_position=lambda: MountPositions.LEFT_MOUNT,
            get_id=lambda: "p1",
            get_pipette=lambda: "P",
        )
        self.flex.available_labware[DeckLocations.A1] = lw
        self.flex.gantry[MountPositions.LEFT_MOUNT] = pip
        # should not raise
        self.flex.validate_configuration(labware=lw, pipette=pip)

    def test_validate_configuration_invalid_labware(self):
        with self.assertRaises(Exception):
            self.flex.validate_configuration(labware=None, pipette=None)

    def test_validate_configuration_invalid_pipette(self):
        lw = MagicMock(
            get_location=lambda: DeckLocations.A1, get_display_name=lambda: "L"
        )
        pip = MagicMock(
            get_mount_position=lambda: MountPositions.LEFT_MOUNT,
            get_id=lambda: "wrong",
            get_pipette=lambda: "P",
        )
        self.flex.available_labware[DeckLocations.A1] = lw
        with self.assertRaises(Exception):
            self.flex.validate_configuration(labware=lw, pipette=pip)


if __name__ == "__main__":
    unittest.main()
