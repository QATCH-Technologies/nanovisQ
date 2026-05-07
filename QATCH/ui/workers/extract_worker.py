import os
import pyzipper
from PyQt5 import QtCore


class ExtractWorker(QtCore.QThread):
    label_text = QtCore.pyqtSignal(str)
    set_range = QtCore.pyqtSignal(int, int)
    progress = QtCore.pyqtSignal(int)
    finished = QtCore.pyqtSignal()

    def __init__(self, save_to, new_install_path):
        super().__init__()
        self.save_to = save_to
        self.new_install_path = new_install_path

    def run(self):
        save_to = self.save_to
        new_install_path = self.new_install_path
        zip_filename = os.path.basename(save_to)[:-4]

        if os.path.basename(new_install_path) != zip_filename:
            new_install_path = os.path.join(new_install_path, zip_filename)

        # Extract ZIP and launch new build
        with pyzipper.AESZipFile(save_to, "r") as zf:
            file_list = zf.namelist()
            total = len(file_list)
            self.set_range.emit(0, total)

            for i, file in enumerate(file_list):
                zf.extract(file, new_install_path)
                self.label_text.emit(f"Extracting: {file}")
                self.progress.emit(i)

        self.label_text.emit("Finalizing...")
        nested_path_wrong = os.path.join(new_install_path, zip_filename)

        if os.path.exists(nested_path_wrong):
            # this guarantees files are extracted where we want them
            os.renames(nested_path_wrong, new_install_path + "_temp")
            if os.path.dirname(save_to) == new_install_path:
                os.renames(
                    save_to,
                    os.path.join(new_install_path + "_temp", zip_filename + ".zip"),
                )
            os.renames(new_install_path + "_temp", new_install_path)

        os.remove(save_to)
        self.finished.emit()
