import multiprocessing
import os
import requests
from PyQt5 import QtCore
from QATCH.common.logger import Logger as Log


class UpdaterProcess_Dbx(multiprocessing.Process):
    def __init__(self, dbx_conn, local, remote):
        super().__init__()
        self._dbx_conn = dbx_conn
        self._local = local
        self._remote = remote

    def run(self):
        self.progressTask(self._local, self._remote)

    def progressTask(self, file_local, file_remote):
        md = self._dbx_conn.files_download_to_file(file_local, file_remote)


class UpdaterTask(QtCore.QThread):
    TAG = "[UpdaterTask]"
    finished = QtCore.pyqtSignal()
    exception = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(str, int)
    _cancel = False

    def __init__(self, local_file, remote_file, total_size):
        super().__init__()
        self.local_file = local_file
        self.remote_file = remote_file
        self.total_size = total_size

    def cancel(self):
        Log.d("GUI: Toggle progress mode")
        # Log.w("Process kill request")
        self._cancel = True

    def run(self):
        raise NotImplementedError("Subclass must define a custom run() method")


class UpdaterTask_Dbx(UpdaterTask):
    def __init__(self, local_file, remote_file, total_size, dbx_conn):
        super().__init__(local_file, remote_file, total_size)
        self._dbx_connection = dbx_conn

    def cancel(self):
        super().cancel()
        self.progressTaskHandle.kill()

    def run(self):
        try:
            save_to = self.local_file
            path = self.remote_file
            size = self.total_size
            last_pct = -1

            self.progressTaskHandle = UpdaterProcess_Dbx(self._dbx_connection, save_to, path)
            self.progressTaskHandle.start()

            while True:
                try:
                    curr_size = os.path.getsize(save_to)
                except FileNotFoundError as e:
                    curr_size = 0
                except Exception as e:
                    curr_size = 0
                    Log.e(self.TAG, f"ERROR: {e}")
                    self.exception.emit(str(e))
                pct = int(100 * curr_size / size)
                if pct != last_pct or curr_size == size:
                    status_str = f"Download Progress: {curr_size} / {size} bytes ({pct}%)"
                    if curr_size == 0:
                        status_str = f"Starting Download: {os.path.basename(path)} ({pct}%)"
                    Log.i(self.TAG, status_str)
                    self.progress.emit(status_str[: status_str.rfind(" (")], pct)
                    need_repaint = True
                    last_pct = pct
                if curr_size == size or self._cancel or not self.progressTaskHandle.is_alive():
                    break
            if not self._cancel:
                Log.d("GUI: Toggle progress mode")
                Log.i(self.TAG, "Finshed downloading!")
            else:
                self.progressTaskHandle.join()
                if os.path.exists(save_to):
                    Log.d(f"Removing partial file download: {save_to}")
                    os.remove(save_to)
        except Exception as e:
            Log.e(self.TAG, f"ERROR: {e}")
            self.exception.emit(str(e))
        self.finished.emit()


class UpdaterTask_Git(UpdaterTask):
    def run(self):
        try:
            save_to = self.local_file
            path = self.remote_file
            size = self.total_size
            last_pct = -1

            status_str = f"Starting Download: {os.path.basename(path)} (0%)"
            Log.i(self.TAG, status_str)
            self.progress.emit(status_str[: status_str.rfind(" (")], 0)

            file_remote = path
            file_local = save_to
            with requests.get(file_remote, stream=True) as r:
                r.raise_for_status()
                size = int(r.headers["Content-Length"])
                with open(file_local, "wb") as f:
                    curr_size = 0
                    for chunk in r.iter_content(chunk_size=8192):
                        if self._cancel:
                            break

                        f.write(chunk)
                        curr_size += len(chunk)

                        pct = int(100 * curr_size / size)
                        if pct != last_pct or curr_size == size:
                            status_str = f"Download Progress: {curr_size} / {size} bytes ({pct}%)"
                            Log.i(self.TAG, status_str)
                            self.progress.emit(status_str[: status_str.rfind(" (")], pct)
                            last_pct = pct

            if not self._cancel:
                Log.d("GUI: Toggle progress mode")
                Log.i(self.TAG, "Finshed downloading!")
            else:
                if os.path.exists(save_to):
                    Log.d(f"Removing partial file download: {save_to}")
                    os.remove(save_to)
        except Exception as e:
            Log.e(self.TAG, f"ERROR: {e}")
            self.exception.emit(str(e))
        self.finished.emit()
