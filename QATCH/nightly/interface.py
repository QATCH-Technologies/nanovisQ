from QATCH.common.logger import Logger as Log
from QATCH.nightly.artifacts import GH_Artifacts
from QATCH.ui.mainWindow import UpdaterTask
from QATCH.ui.popUp import PopUp

from threading import Thread, Event

import os


class GH_Interface:

    def __init__(self, parent) -> None:
        self.artifacts = GH_Artifacts()
        self.parent = parent

    def update_check(self) -> tuple[tuple[bool, bool], tuple[dict, str]]:
        update_result = None
        labelweb2 = "OFFLINE"
        labelweb3 = "UNKNOWN"

        try:
            update_result = self.artifacts.check_for_update()
            labelweb2 = "ONLINE"
        except:
            Log.e("HTTP Error, assuming we must be offline!")

        if update_result is None:
            labelweb3 = "UP-TO-DATE!"
        else:
            labelweb3 = f"{update_result[1]['name'].split('_')[-1]} available!"

        Log.i("[NightBuild]",
              'Checking your internet connection {} '.format(labelweb2))
        Log.i("Update Status:", labelweb3)

        update_available = False
        update_now = False
        latest_bundle = None

        if update_result is not None:
            update_available = True
            running, latest, key = update_result
            latest_bundle = (latest, key)
            if PopUp.question_FW(self.parent,
                                 "QATCH Update Available!",
                                 "A new nightly software build is available!\nWould you like to download it now?",
                                 "Running SW: {}\nRecommended: {}\n".format(
                                     running['name'].split('_')[-1],
                                     latest['name'].split('_')[-1]) +
                                 "Created: {}\n\nPlease save your work before updating.".format(
                                     str(latest['created_at']).replace("T", " @ ").replace("Z", " UTC"))):
                update_now = True

        return ((update_available, update_now), latest_bundle)


class UpdaterTask_Nightly(UpdaterTask):

    def __init__(self, local_file, remote_file, total_size):
        super().__init__(local_file, remote_file, total_size)
        self.abort_flag = Event()

    def cancel(self):
        super().cancel()
        self.abort_flag.set()

    def run(self):
        try:
            save_to = self.local_file

            artifacts = GH_Artifacts()
            update_result = artifacts.check_for_update()
            if update_result is None:
                raise FileNotFoundError("No nightly builds found.")

            running, latest, key = update_result
            latest_bundle = (latest, key)

            self.progressTaskHandle = Thread(
                target=artifacts.prepare_for_update,
                args=(latest_bundle, save_to, self.progress, self.abort_flag))
            self.progressTaskHandle.start()

            while True:
                self.progressTaskHandle.join()  # wait for exit
                if self._cancel or not self.progressTaskHandle.is_alive():
                    break
            if not self._cancel:
                Log.d("GUI: Toggle progress mode")
                Log.i(self.TAG, "Finshed downloading!")
                artifacts.delete_latest_build_file()
            else:
                self.progressTaskHandle.join()
                if os.path.exists(save_to):
                    Log.d(f"Removing partial file download: {save_to}")
                    os.remove(save_to)

        except Exception as e:
            Log.e(self.TAG, f"ERROR: {e}")
            self.exception.emit(str(e))
        self.finished.emit()
