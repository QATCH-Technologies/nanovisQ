from functools import partial as callback_args
from pathlib import Path
from queue import Queue, Empty
from subprocess import Popen, PIPE, STDOUT
from threading import Thread
from PyQt5.QtCore import pyqtSignal
from QATCH.common.architecture import Architecture
from QATCH.common.logger import Logger as Log
from os import system, path


class QATCH_TyUpdater:

    # NOTE: It is desirable to still send "PROGRAM" command to device prior to flashing
    #       Best practice: Wait until you receive "Waiting for lines..." before reboot.

    def __init__(self, progress_signal=None):
        if progress_signal == None:
            progress_signal = pyqtSignal(str, int)
        self.progress = progress_signal

    def update(self, device_sernum, firmware_path):

        queue_timeout = 5  # seconds of no subprocess.stdout bytes to trigger a timeout exception
        tools_path = path.join(Architecture.get_path(), "tools")
        path_to_tycmd = path.join(tools_path, "tytools", "tycmd.exe")
        path_to_loader = path.join(
            tools_path, "tool-teensy", "teensy_loader_cli.exe")
        teensy_mcu = "TEENSY41"

        Log.d("Path to TyCmd: ", path_to_tycmd)
        Log.d("Path to Loader:", path_to_loader)

        result_error = "ERROR: Something unexpected occurred!"

        def reader(proc, queue):
            try:
                source = Path(proc.args[0]).stem
                with proc.stdout as pipe:
                    while True:
                        byte = pipe.read1(1)
                        # if source == "teensy_loader_cli" and byte == b")":
                        #     print('hehe', end='\r')
                        if len(byte) == 1:
                            queue.put((source, byte))
                        else:
                            break
            finally:
                queue.put(None)

        cmd_updater = f'{path_to_tycmd}  upload  --board  {device_sernum}  {firmware_path}'.split(
            '  ')
        cmd_restart = f'{path_to_tycmd}  reset  --board  {device_sernum}  --bootloader'.split(
            '  ')
        cmd_flasher = f'{path_to_loader}  --mcu={teensy_mcu}  -v  {firmware_path}  -w'.split(
            '  ')

        p_updater, p_flasher, p_restart = None, None, None
        t_updater, t_flasher, t_restart = None, None, None
        pass_restart, pass_flasher = False, False

        try:
            # protection to force kill programmer that will brick a unit if left open
            print("NOTE: You can safely ignore if the next line is an ERROR")
            system("taskkill /im teensy.exe")

            q = Queue()

            p_flasher = Popen(cmd_flasher, stdout=PIPE,
                              stderr=STDOUT, shell=True)
            t_flasher = Thread(target=reader, args=[p_flasher, q])
            t_flasher.daemon = True
            t_flasher.start()

            p_restart = Popen(cmd_restart, stdout=PIPE,
                              stderr=STDOUT, shell=True)
            t_restart = Thread(target=reader, args=[p_restart, q])
            t_restart.daemon = True
            t_restart.start()

            line_restart, line_flasher, = b"", b""
            total_pages, num_pages, curr_pct = 0, 0, 0
            last_pct = -1

            for i in range(3):
                if i == 2 and p_updater == None:
                    break  # do not wait for 3rd queue
                for source, byte in iter(callback_args(q.get, timeout=queue_timeout), None):
                    # print("%s: %s" % (source, data))
                    if source == Path(cmd_restart[0]).stem:
                        line_restart += byte
                        if byte == b'\n':
                            most_recent_line = line_restart.splitlines()[-1]
                            print("%s: %s" % (source, most_recent_line))
                            if most_recent_line.find(b"Board is already in bootloader mode") >= 0:
                                print(
                                    "Warning: Board is already in bootloader mode")
                                pass_restart = True
                            if most_recent_line.find(b"Triggering board reboot") >= 0:
                                print(
                                    "SUCCESS: Board reset and entering bootloader...")
                                pass_restart = True
                    if source == Path(cmd_flasher[0]).stem:
                        line_flasher += byte
                        if byte == b'\n':
                            most_recent_line = line_flasher.splitlines()[-1]
                            print("%s: %s" % (source, most_recent_line))
                            if most_recent_line.startswith(b"Read"):
                                total_pages = int(
                                    most_recent_line.split(b' ')[-4]) / 1024
                            if most_recent_line.startswith(b"Booting"):
                                pass_flasher = True
                            if most_recent_line.find(b"error writing to Teensy") >= 0 and p_updater == None:
                                # try again, no wait (likely pass)
                                p_updater = Popen(
                                    cmd_flasher[:-1], stdout=PIPE, stderr=STDOUT, shell=True)
                                t_updater = Thread(
                                    target=reader, args=[p_updater, q])
                                t_updater.daemon = True
                                t_updater.start()
                        if byte == b'.':
                            most_recent_line = line_flasher.splitlines()[-1]
                            if most_recent_line.startswith(b"Programming"):
                                num_pages = most_recent_line.count(b".")
                                curr_pct = int(100 * num_pages / total_pages)
                                if curr_pct == 0:
                                    starting = True
                                    curr_txt = "Starting firmware transfer..."
                                else:
                                    if starting:
                                        print()
                                    starting = False
                                    # curr_txt = f"b'Programming {curr_pct}%'"
                                    curr_txt = "Programming device firmware...<br/><b>DO NOT POWER CYCLE DEVICE!</b>"
                                # Log.d("%s: %s" % (source, curr_txt))
                                if curr_pct != last_pct:
                                    self.progress.emit(curr_txt, curr_pct)
                                    last_pct = curr_pct

        except Empty:
            result_error = "ERROR: A timeout occurred while reading from the queue."
            Log.e(result_error)

        except Exception as e:
            result_error = "ERROR: An unhandled exception occurred!"
            Log.e(result_error)
            Log.e(e)

        finally:
            if p_flasher != None and p_flasher.poll() == None:
                # subprocess still running, kill it on timeout
                p_flasher.kill()
            if p_restart != None and p_restart.poll() == None:
                # subprocess still running, kill it on timeout
                p_restart.kill()
            if p_updater != None and p_updater.poll() == None:
                # subprocess still running, kill it on timeout
                p_updater.kill()

        if pass_flasher and pass_restart:
            result_error = ""  # success
            Log.i("SUCCESS: Updated successfully")
            if p_updater == None:
                Log.d("BONUS: No retry was required!")
        else:
            # on failure, exit bootloader to return to main application...
            Popen(cmd_restart[:-1], shell=True)
            if not pass_flasher:
                result_error = "ERROR: Failed to program."
                Log.e(result_error)
            if not pass_restart:
                result_error = "ERROR: Failed to reboot."
                Log.e(result_error)

        if p_restart != None:
            t_restart.join()
            p_restart.wait()

        if p_flasher != None:
            t_flasher.join()
            p_flasher.wait()

        if p_updater != None:
            t_updater.join()
            p_updater.wait()

        Log.d("TyUpdater Task gracefully finished.")
        return result_error
