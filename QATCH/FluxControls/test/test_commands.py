# test_commands.py
import unittest
from unittest.mock import patch, MagicMock

from src.commands import Commands, CommandType, Intents, Axis, DeckLocations, SlotName
from src.pipette import Pipette
from src.labware import Labware
from src.constants import Pipettes, MountPositions
from src.standard_labware import StandardWellplates


class TestCommands(unittest.TestCase):
    def test_create_base_command(self):
        payload = Commands._create_base_command(
            CommandType.ASPIRATE,
            {"foo": "bar"},
            Intents.SETUP
        )
        self.assertIn("data", payload)
        data = payload["data"]
        self.assertEqual(data["commandType"], CommandType.ASPIRATE.value)
        self.assertEqual(data["intent"], Intents.SETUP.value)
        self.assertEqual(data["params"], {"foo": "bar"})

    def test_load_labware(self):
        loc = DeckLocations.A1
        payload = Commands.load_labware(loc, "plate_96_well", "opentrons", 1)
        data = payload["data"]
        self.assertEqual(data["commandType"], CommandType.LOAD_LABWARE.value)
        self.assertEqual(data["intent"], Intents.SETUP.value)
        self.assertEqual(
            data["params"],
            {
                "location": SlotName.get_slot_name(loc),
                "loadName": "plate_96_well",
                "namespace": "opentrons",
                "version": 1
            }
        )

    def test_load_pipette(self):
        pip = Pipette(pipette=Pipettes.P50_MULTI_FLEX,
                      mount_position=MountPositions.LEFT_MOUNT)
        payload = Commands.load_pipette(pip)
        data = payload["data"]
        self.assertEqual(data["commandType"], CommandType.LOAD_PIPETTE.value)
        self.assertEqual(data["intent"], Intents.SETUP.value)
        self.assertEqual(
            data["params"],
            {"pipetteName": pip.pipette, "mount": pip.mount_position}
        )

    def test_pickup_tip(self):
        lab = Labware(location=DeckLocations.A1,
                      labware_definition=StandardWellplates.APPLIED_BIOSYSTEMS_MICROAMP_384_WELLPLATE_40UL)
        pip = Pipette(pipette=Pipettes.P50_MULTI_FLEX,
                      mount_position=MountPositions.LEFT_MOUNT)
        payload = Commands.pickup_tip(lab, pip)
        data = payload["data"]
        self.assertEqual(data["commandType"], CommandType.PICKUP_TIP.value)
        self.assertEqual(data["intent"], Intents.SETUP.value)
        self.assertEqual(
            data["params"],
            {
                "labwareId": lab.id,
                "wellName": lab.location,
                "wellLocation": {"origin": "top", "offset": lab.get_offsets()},
                "pipetteId": pip.id
            }
        )

    def test_aspirate_and_dispense(self):
        lab = Labware(location=DeckLocations.A1,
                      labware_definition=StandardWellplates.APPLIED_BIOSYSTEMS_MICROAMP_384_WELLPLATE_40UL)
        pip = Pipette(pipette=Pipettes.P50_MULTI_FLEX,
                      mount_position=MountPositions.LEFT_MOUNT)
        fr, vol = 10.5, 50.0
        for method, ctype in (
            (Commands.aspirate, CommandType.ASPIRATE),
            (Commands.dispense, CommandType.DISPENSE)
        ):
            payload = method(lab, pip, fr, vol)
            data = payload["data"]
            self.assertEqual(data["commandType"], ctype.value)
            self.assertEqual(data["intent"], Intents.SETUP.value)
            self.assertEqual(
                data["params"],
                {
                    "labwareId": lab.id,
                    "wellName": lab.location,
                    "wellLocation": {"origin": "top", "offset": lab.get_offsets()},
                    "flowRate": fr,
                    "volume": vol,
                    "pipetteId": pip.id
                }
            )

    def test_blowout_and_drop_tip(self):
        lab = Labware(location=DeckLocations.A1,
                      labware_definition=StandardWellplates.APPLIED_BIOSYSTEMS_MICROAMP_384_WELLPLATE_40UL)
        pip = Pipette(pipette=Pipettes.P50_MULTI_FLEX,
                      mount_position=MountPositions.LEFT_MOUNT)
        fr = 20.0
        blow = Commands.blowout(lab, pip, fr)
        drop = Commands.drop_tip(lab, pip)

        # blowout
        b = blow["data"]
        self.assertEqual(b["commandType"], CommandType.BLOWOUT.value)
        self.assertEqual(
            b["params"],
            {
                "labwareId": lab.id,
                "wellName": lab.location,
                "wellLocation": {"origin": "top", "offset": lab.get_offsets()},
                "flowRate": fr,
                "pipetteId": pip.id
            }
        )
        # drop tip
        d = drop["data"]
        self.assertEqual(d["commandType"], CommandType.DROP_TIP.value)
        self.assertEqual(
            d["params"],
            {
                "labwareId": lab.id,
                "wellName": lab.location,
                "wellLocation": {"origin": "top", "offset": lab.get_offsets()},
                "pipetteId": pip.id
            }
        )

    def test_move_to_well(self):
        lab = Labware(location=DeckLocations.A1,
                      labware_definition=StandardWellplates.APPLIED_BIOSYSTEMS_MICROAMP_384_WELLPLATE_40UL)
        pip = Pipette(pipette=Pipettes.P50_MULTI_FLEX,
                      mount_position=MountPositions.LEFT_MOUNT)
        payload = Commands.move_to_well(lab, pip)
        data = payload["data"]
        self.assertEqual(data["commandType"], CommandType.MOVE_TO_WELL.value)
        self.assertEqual(
            data["params"],
            {
                "labwareId": lab.id,
                "wellName": lab.location,
                "wellLocation": {"origin": "top", "offset": lab.get_offsets()},
                "pipetteId": pip.id
            }
        )

    def test_move_to_coordinates(self):
        pip = Pipette(pipette=Pipettes.P50_MULTI_FLEX,
                      mount_position=MountPositions.LEFT_MOUNT)
        payload = Commands.move_to_coordinates(pip, 1.0, 2.0, 3.0, 5.0, True)
        data = payload["data"]
        self.assertEqual(data["commandType"], CommandType.MOVE_TO_WELL.value)
        self.assertEqual(
            data["params"],
            {
                "coordinates": {"x": 1.0, "y": 2.0, "z": 3.0},
                "minimumZHeight": 5.0,
                "forceDirect": True,
                "pipetteId": pip.id
            }
        )

    def test_move_relative(self):
        pip = Pipette(pipette=Pipettes.P50_MULTI_FLEX,
                      mount_position=MountPositions.LEFT_MOUNT)
        payload = Commands.move_relative(pip, 7.5, Axis.Z)
        data = payload["data"]
        self.assertEqual(data["commandType"], CommandType.MOVE_TO_WELL.value)
        self.assertEqual(
            data["params"],
            {"axis": Axis.Z.value, "distance": 7.5, "pipetteId": pip.id}
        )

    @patch('commands.requests.post')
    def test_send_command_success(self, mock_post):
        # simulate OK response
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"status": "success"}
        mock_post.return_value = mock_resp

        result = Commands.send_command("http://robot/commands", {"data": {}})
        mock_post.assert_called_once()
        self.assertEqual(result, {"status": "success"})

    @patch('commands.requests.post')
    def test_send_command_failure(self, mock_post):
        # simulate network error
        mock_post.side_effect = Exception("boom")
        result = Commands.send_command("http://robot/commands", {"data": {}})
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
