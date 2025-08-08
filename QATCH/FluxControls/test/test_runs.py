# test_runs.py
import json
import unittest
import io
import inspect
from unittest.mock import patch, MagicMock
import requests

from src.runs import Runs, Actions, Lights


class FakeFile:

    def __init__(self):
        self.closed = False

    def read(self):
        return b""

    def close(self):
        self.closed = True


class TestRuns(unittest.TestCase):
    def setUp(self):
        # Patch HEADERS in the module so _send_request always has something to pass along
        self.module = inspect.getmodule(Runs)
        self._orig_headers = getattr(self.module, "HEADERS", None)
        setattr(self.module, "HEADERS", {"Authorization": "test-token"})

    def tearDown(self):
        # restore original HEADERS
        if self._orig_headers is not None:
            setattr(self.module, "HEADERS", self._orig_headers)

    # ----------------------------
    # _send_request tests
    # ----------------------------
    def test_send_request_post_with_json(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"ok": True}

        with patch("requests.post", return_value=mock_resp) as mock_post:
            result = Runs._send_request("POST", "http://example.com", {"a": 1})
            mock_post.assert_called_once_with(
                url="http://example.com",
                headers={"Authorization": "test-token"},
                data=json.dumps({"a": 1}),
            )
            self.assertEqual(result, {"ok": True})

    def test_send_request_post_with_files_dict(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"uploaded": True}

        files_payload = {"files": io.BytesIO(b"")}
        with patch("requests.post", return_value=mock_resp) as mock_post:
            result = Runs._send_request(
                "POST", "http://example.com", files_payload)
            mock_post.assert_called_once_with(
                url="http://example.com",
                headers={"Authorization": "test-token"},
                files=files_payload,
            )
            self.assertEqual(result, {"uploaded": True})

    def test_send_request_post_with_files_list(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"multiple": True}

        list_payload = [("files", io.BytesIO(b"")), ("files", io.BytesIO(b""))]
        with patch("requests.post", return_value=mock_resp) as mock_post:
            result = Runs._send_request(
                "POST", "http://example.com", list_payload)
            mock_post.assert_called_once_with(
                url="http://example.com",
                headers={"Authorization": "test-token"},
                files=list_payload,
            )
            self.assertEqual(result, {"multiple": True})

    def test_send_request_get(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"data": [1, 2, 3]}

        with patch("requests.get", return_value=mock_resp) as mock_get:
            result = Runs._send_request("GET", "http://example.com")
            mock_get.assert_called_once_with(
                url="http://example.com",
                headers={"Authorization": "test-token"},
            )
            self.assertEqual(result, {"data": [1, 2, 3]})

    def test_send_request_delete(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"deleted": True}

        with patch("requests.delete", return_value=mock_resp) as mock_delete:
            result = Runs._send_request("DELETE", "http://example.com")
            mock_delete.assert_called_once_with(
                url="http://example.com",
                headers={"Authorization": "test-token"},
            )
            self.assertEqual(result, {"deleted": True})

    def test_send_request_unsupported_method(self):
        with self.assertRaises(ValueError):
            Runs._send_request("PUT", "http://example.com")

    def test_send_request_network_error(self):
        with patch("requests.post", side_effect=requests.exceptions.RequestException("boom")):
            result = Runs._send_request("POST", "http://example.com", {"a": 1})
            self.assertIsNone(result)

    # ----------------------------
    # run_protocol & delete_protocol
    # ----------------------------
    def test_run_protocol_returns_id(self):
        with patch.object(Runs, "_send_request", return_value={"data": {"id": "run123"}}) as mock_sr:
            run_id = Runs.run_protocol("http://runs", "protABC")
            mock_sr.assert_called_once_with(
                "POST",
                "http://runs",
                {"data": {"protocolId": "protABC"}},
            )
            self.assertEqual(run_id, "run123")

    def test_delete_protocol_builds_url(self):
        with patch.object(Runs, "_send_request", return_value={"ok": True}) as mock_sr:
            result = Runs.delete_protocol("http://prots", "p1")
            mock_sr.assert_called_once_with("DELETE", "http://prots/p1")
            self.assertEqual(result, {"ok": True})

    # ----------------------------
    # upload_protocol
    # ----------------------------
    def test_upload_protocol_opens_and_closes_file(self):
        # Prepare a fake file and patch open
        fake_file = FakeFile()

        def fake_open(path, mode):
            self.assertEqual(path, "mypath")
            self.assertEqual(mode, "rb")
            return fake_file

        with patch(f"{self.module.__name__}.open", fake_open), \
                patch.object(Runs, "_send_request", return_value={
                    "data": {"id": "pid", "metadata": {"protocolName": "myProt"}}
                }) as mock_sr:
            out = Runs.upload_protocol("http://prots", "mypath")
            # ensure file was closed
            self.assertTrue(fake_file.closed)
            self.assertEqual(out, {"protocol_id": "pid",
                             "protocol_name": "myProt"})
            mock_sr.assert_called_once()

    # ----------------------------
    # upload_protocol_custom_labware
    # ----------------------------
    def test_upload_protocol_custom_labware_closes_all(self):
        # three FakeFiles: one for protocol, two for labware
        f_proto, f_lab1, f_lab2 = FakeFile(), FakeFile(), FakeFile()
        opens = [f_proto, f_lab1, f_lab2]

        def fake_open(path, mode):
            # pop next fake file
            return opens.pop(0)

        with patch(f"{self.module.__name__}.open", fake_open), \
                patch.object(Runs, "_send_request", return_value={
                    "data": {"id": "X", "metadata": {"protocolName": "Y"}}
                }) as mock_sr:
            out = Runs.upload_protocol_custom_labware(
                "http://prots", "protPath", ["lab1", "lab2"]
            )
            # all three should be closed
            self.assertTrue(f_proto.closed)
            self.assertTrue(f_lab1.closed)
            self.assertTrue(f_lab2.closed)
            self.assertEqual(out, {"protocol_id": "X", "protocol_name": "Y"})
            mock_sr.assert_called_once()

    # ----------------------------
    # delete_run
    # ----------------------------
    def test_delete_run(self):
        with patch.object(Runs, "_send_request", return_value={"ok": True}) as mock_sr:
            res = Runs.delete_run("http://runs", 42)
            mock_sr.assert_called_once_with("DELETE", "http://runs/42")
            self.assertEqual(res, {"ok": True})

    # ----------------------------
    # list & status retrieval
    # ----------------------------
    def test_get_protocols_list(self):
        proto_list = [{"id": 1}, {"id": 2}]
        with patch.object(Runs, "_send_request", return_value={"data": proto_list}) as mock_sr:
            out = Runs.get_protocols_list("http://prots")
            mock_sr.assert_called_once_with("GET", "http://prots")
            self.assertEqual(out, proto_list)

    def test_get_run_status(self):
        status = {"state": "running"}
        with patch.object(Runs, "_send_request", return_value=status) as mock_sr:
            out = Runs.get_run_status("http://runs", 99)
            mock_sr.assert_called_once_with("GET", "http://runs/99")
            self.assertEqual(out, status)

    def test_get_runs_list(self):
        runs = [{"id": "a"}, {"id": "b"}]
        with patch.object(Runs, "_send_request", return_value={"data": runs}) as mock_sr:
            out = Runs.get_runs_list("http://runs")
            mock_sr.assert_called_once_with("GET", "http://runs")
            self.assertEqual(out, runs)

    # ----------------------------
    # run control (pause/play/stop/resume)
    # ----------------------------
    def _assert_action(self, method_name, action_enum):
        """Helper to test pause/play/stop/resume"""
        runs_url = "http://runs"
        run_id = "RID"
        expected_url = f"{runs_url}/{run_id}/actions"
        expected_payload = {"data": {"actionType": action_enum.value}}

        with patch.object(Runs, "_send_request", return_value={"ok": True}) as mock_sr:
            method = getattr(Runs, method_name)
            out = method(runs_url, run_id)
            mock_sr.assert_called_once_with(
                "POST", expected_url, expected_payload)
            self.assertEqual(out, {"ok": True})

    def test_pause_run(self):
        self._assert_action("pause_run", Actions.PAUSE)

    def test_play_run(self):
        self._assert_action("play_run", Actions.PLAY)

    def test_stop_run(self):
        self._assert_action("stop_run", Actions.STOP)

    def test_resume_run(self):
        self._assert_action("resume_run", Actions.RESUME)

    def test_set_lights(self):
        status = Lights.ON
        with patch.object(Runs, "_send_request", return_value={"ok": True}) as mock_sr:
            out = Runs.set_lights("http://lights", status)
            mock_sr.assert_called_once_with("POST", "http://lights", status)
            self.assertEqual(out, {"ok": True})

    def test_get_lights(self):
        resp = {"state": "lit"}
        with patch.object(Runs, "_send_request", return_value=resp) as mock_sr:
            out = Runs.get_lights("http://lights")
            mock_sr.assert_called_once_with(method="GET", url="http://lights")
            self.assertEqual(out, resp)

    def test_home(self):
        home_resp = {"homed": True}
        with patch.object(Runs, "_send_request", return_value=home_resp) as mock_sr:
            out = Runs.home("http://home")
            expected_payload = json.dumps({"target": "robot"})
            mock_sr.assert_called_once_with(
                method="POST",
                url="http://home",
                payload=expected_payload
            )
            self.assertEqual(out, home_resp)


if __name__ == "__main__":
    unittest.main()
