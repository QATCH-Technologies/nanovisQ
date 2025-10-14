import os
import shutil

from QATCH.core.constants import Constants
sw_version = Constants.app_version
fw_version = Constants.best_fw_version

from_path = os.path.expanduser("~/Downloads")
to_path = os.path.expanduser("~/Documents/QATCH Work/v2.6x branch/trunk")
pattern = "QATCH Release Notes"


def move_docs():

    files = [f for f in os.listdir(from_path) if f.startswith(pattern)]
    print(files)

    if len(files) != 3:
        print(f"Expecting 3 files, found {len(files)}. No changes made.")
        print("Press enter key to close.")
        input()
        return  # abort

    for file in files:
        full_path = os.path.join(from_path, file)
        sheet = file.split("-")[-1].strip()
        if "FW Changes" in sheet:
            move_fw_changes(full_path)
        elif "Release Notes" in sheet:
            move_release_notes(full_path)
        elif "SW Changes" in sheet:
            move_sw_changes(full_path)
        else:
            print(f"Unknown file: {file}")


def move_fw_changes(path):

    dst1 = os.path.join(to_path, "QATCH_Q-1_FW_py_dev",
                        "docs", "FW Change Control Doc.pdf")
    if os.path.exists(dst1):
        os.remove(dst1)
    else:
        print("FW Changes file was missing. Is this the right place?")
        print(dst1)
    shutil.copy(path, dst1)  # copy first, move later

    dst2 = os.path.join(
        to_path, f"QATCH_Q-1_FW_py_{fw_version}", "FW Change Control Doc.pdf")
    if os.path.exists(dst2):
        os.remove(dst2)
    else:
        print("FW Changes file was missing. Is this the right place?")
        print(dst2)
    shutil.move(path, dst2)  # move
    print("Moved FW Changes")


def move_release_notes(path):

    dst = os.path.join(
        to_path, "docs", f"Release Notes {sw_version}.pdf")
    files = [f for f in os.listdir(os.path.dirname(
        dst)) if f.startswith("Release Notes")]
    if len(files) == 1:
        full_path = os.path.join(os.path.dirname(dst), files[0])
        os.remove(full_path)
    else:
        print("Release Notes file was missing. Is this the right place?")
        print(os.path.dirname(dst))
    shutil.move(path, dst)  # move
    print("Moved Release Notes")


def move_sw_changes(path):

    dst = os.path.join(
        to_path, "QATCH", "SW Change Control Doc.pdf")
    if os.path.exists(dst):
        os.remove(dst)
    else:
        print("SW Changes file was missing. Is this the right place?")
        print(dst)
    shutil.move(path, dst)  # move
    print("Moved SW Changes")


move_docs()
