# Device.py
# A dynamic encapsulator for 'serial' and 'requests' modules
# Assuming it is emulating an Arduino (or similar) device.

from QATCH.common.logger import Logger as Log
from QATCH.common.findDevices import Discovery
from time import time
from struct import pack
from serial.tools import list_ports
import random
import numpy as np
import serial as sp
import requests
import ctypes
import sys
import os

try:
    if getattr(sys, 'frozen', False):
        # we are running in a bundle: use "_MEIPASS" folder
        bundle_dir = sys._MEIPASS
    else:
        # we are running in a normal Python environment: use "QATCH\.libs" folder
        bundle_dir = os.path.join(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))), ".libs")
    # define a known position for this DLL, for both EXE and PY code
    hidapi_dll_path = os.path.join(bundle_dir, "hidapi", "hidapi.dll")
    ctypes.CDLL(hidapi_dll_path)
    import hid
    HID_DLL_FOUND = True
except Exception as e:
    Log.e("ERROR: " + str(e))
    Log.e("Please install and/or locate libhid: missing 'hidapi.dll'")
    HID_DLL_FOUND = False

TAG = "[Device]"

# a Serial class emulator


class serial:

    STOPBITS_ONE = 1
    EIGHTBITS = 8
    HID_TX_SIZE = 64
    HID_RX_SIZE = 64
    HID_RX_INTERVAL = 100

    # init(): the constructor.  Many of the arguments have default values
    # and can be skipped when calling the constructor.
    def Serial(port='COM1', baudrate=19200, timeout=1,
               bytesize=8, parity='N', stopbits=1, xonxoff=0,
               rtscts=0, dsrdtr=0, write_timeout=None,
               inter_byte_timeout=None):

        return serial(port, baudrate, timeout,
                      bytesize, parity, stopbits, xonxoff,
                      rtscts, dsrdtr, write_timeout,
                      inter_byte_timeout)

    def __init__(self, port='COM1', baudrate=19200, timeout=1,
                 bytesize=8, parity='N', stopbits=1, xonxoff=0,
                 rtscts=0, dsrdtr=0, write_timeout=None,
                 inter_byte_timeout=None):

        self._serial = sp.Serial()

        # public set/get params
        self.port = port
        self.timeout = timeout
        self.parity = parity
        self.baudrate = baudrate
        self.bytesize = bytesize
        self.stopbits = stopbits
        self.xonxoff = xonxoff
        self.rtscts = rtscts
        self.dsrdtr = dsrdtr
        self.write_timeout = write_timeout
        self.inter_byte_timeout = inter_byte_timeout

        # private set; public get
        self._is_open = False
        self._in_waiting = 0
        self._device = None
        self._hid_buffer = b""
        # self._hid_finished = False

    @staticmethod
    def enumerate(_com: bool | None = None,
                  _net: bool | None = None,
                  _hid: bool | None = None):

        default_to = True
        if True in [_com, _net, _hid]:
            default_to = False
        if _com == None:
            _com = default_to
        if _net == None:
            _net = default_to
        if _hid == None:
            _hid = default_to
        devices = []

        if _com:
            Log.d("Enumerating COM# devices...")
            # perform serial port COM# lookup
            comports_connected = []
            ports_avaiable = list(list_ports.comports())
            for port in ports_avaiable:
                if port[2].startswith("USB VID:PID=16C0:0483"):
                    comports_connected.append(port[0])
            devices.extend(comports_connected)

        if _net:
            Log.d("Enumerating IP addresses...")
            # perform IP address sniff lookup
            nets_connected = Discovery().doDiscover()
            devices.extend(nets_connected)

        if _hid:
            # technically, it's serial emulation over HID, not HIDRAW
            Log.d("Enumerating HIDRAW paths...")
            # perform HID VID/PID path lookup
            hids_connected = []
            if not HID_DLL_FOUND:
                Log.e("Unable to enumerate HIDRAW paths. Missing dependency: hidapi.dll")
            else:
                # Teensy vendor ID and product ID
                vid = 0x16C0
                # NOTE: this is different that for a Teensy serial device (0x0483)
                pid = 0x0486
                for device_dict in hid.enumerate():
                    keys = list(device_dict.keys())
                    keys.sort()

                    if "vendor_id" not in keys or "product_id" not in keys:  # can we do vid/pid check?
                        continue
                    # do vid/pid match?
                    if device_dict["vendor_id"] != vid or device_dict["product_id"] != pid:
                        continue
                    if "product_string" not in keys or "manufacturer_string" not in keys:  # can we get serial number?
                        continue
                    # does manufacturer string match?
                    if device_dict["manufacturer_string"] != "QATCH":
                        continue
                    # does manufacturer string match?
                    if device_dict["product_string"] != "QATCH nanovisQ":
                        continue
                    if "interface_number" not in keys:  # can we determine HIDRAW vs SEREMU?
                        continue
                    if "serial_number" not in keys or "path" not in keys:  # do we have lookup key/path available?
                        continue
                    # is this the Serial Emulator?
                    if device_dict["interface_number"] == 1:
                        hids_connected.append(
                            device_dict["serial_number"] + ':' + device_dict["path"].decode())
            devices.extend(hids_connected)

        Log.d(f"Enumerated {len(devices)} devices on the system.")
        return devices  # return list of found devices

    # isOpen()
    # returns True if the port to the Arduino is open.  False otherwise
    def isOpen(self):
        return self.is_open

    # open()
    # opens the port
    def open(self):
        if self._hid != None:
            _hid_serno, _hid_path = self._hid.split(':')
            try:
                if self._device == None or self._device._Device__dev == None:
                    self._device = hid.Device(path=_hid_path.encode())
                    self._device.nonblocking = 1  # enable non-blocking mode
            except Exception as e:
                Log.e(TAG, f"Failed to open device: {_hid_serno}")
                raise PermissionError("Failed to open device.")
        if self._net != None:
            # issue test request, port 8080 (should be 'ok'; i.e. open)
            try:
                self._request = requests.get(
                    f'http://{self._net}:8080/version', stream=True)
                # return # stop on success; no, open both ports if both are available
            except Exception as e:
                Log.e(TAG, f"Failed to open device: {self._net}")
                if self._request.ok:
                    self._request.status_code = 404
                raise PermissionError("Failed to open device.")
        if self._com != None:
            try:
                # Log.d("COM port{} opened!".format(" already " if self._serial.is_open else ""))
                if not self._serial.is_open:
                    self._serial.open()
                return  # stop on success
            except Exception as e:
                Log.e(TAG, f"Failed to open device: {self._com}")
                # self._com = None # try using ethernet only mode
                raise PermissionError("Failed to open device.")

    # close()
    # closes the port
    def close(self):
        if self._hid != None:
            if self._device != None:
                self._device.close()
        if self._net != None:
            self._request.close()  # stop streaming
            self._request.status_code = 404
        if self._com != None:
            # Log.d("COM port closed!")
            self._serial.close()

    # write()
    # writes a string of characters to the Arduino
    def write(self, string):
        # Log.w("write:", string)
        if self._hid != None:
            bytes_to_send = len(string)
            while bytes_to_send > 0:
                start_from = len(string)-bytes_to_send
                try:
                    if self._device == None:
                        self.open()
                    self._device.write(
                        b"\x00" + string[start_from:start_from+self.HID_TX_SIZE])
                except Exception as e:
                    self.close()
                    # Log.e("HID write error: " + str(e))
                    raise sp.SerialException(e)
                bytes_to_send -= self.HID_TX_SIZE
            TX_TIMEOUT = None if self.write_timeout == None else int(
                self.write_timeout * 1000) if len(self._hid_buffer) == 0 else self.HID_RX_INTERVAL

            self._hid_buffer += self._hid_read_singleshot(TX_TIMEOUT)
            self._in_waiting = len(self._hid_buffer)
            if self._in_waiting > 0 or self.write_timeout == 0:
                # self._hid_finished = False
                return  # stop on success

        if self._net != None:
            cmds = string.decode().strip().split('\n')
            self._peek = b''
            for net_string in cmds:
                replace_sets = [['\n', '&'], [' ', '='], ['?', '_']]
                for set in replace_sets:
                    net_string = net_string.replace(set[0], set[1])
                query_url = f"http://{self._net}:8080/{net_string}"
                try:
                    self._request = requests.get(query_url, stream=True)
                except:
                    if self._request.ok:
                        self._request.status_code = 500
                Log.d(TAG, f"QUERY-{self._request.status_code}: {query_url}")
                if self._request.ok:
                    tmp = self._request.raw.read(1028)
                    self._peek += tmp
                    self._in_waiting += len(tmp)
                else:
                    Log.e(TAG, f"Failed to read device: {self._net}")
                    # fallback to using serial direct (no ethernet)
                    self._net = None
                    break  # continue to sending failed 'string' over _com phy now
            if self._request.ok:
                return  # stop on success

        if self._com != None:
            if not self._serial.is_open:
                self._serial.open()
            self._serial.write(string)
            # wait for reply (with timeout)
            sent_at = time()
            waitFor = 3  # timeout delay (seconds)
            while (time() - sent_at < waitFor and self._serial.in_waiting == 0):
                pass
            if time() - sent_at < waitFor:
                return  # stop on success
            else:
                Log.e(TAG, "ERROR: COM timeout; no reply")

        Log.e(TAG, "ERROR: No available device phy for write().")
        self._in_waiting = 0

    # read()
    # reads n characters from the device hardware. Actually n characters
    # are read from the string _out_data and returned to the caller.
    def read(self, n=-1):
        reply = b''

        if self._hid != None:
            real_bytes_rcvd = 0
            if self.in_waiting <= n or n <= 0:  # not self._hid_finished:
                RX_TIMEOUT = int(
                    self.timeout * 1000) if len(self._hid_buffer) == 0 else self.HID_RX_INTERVAL
                # Log.w("Read timeout:", RX_TIMEOUT)
                this_len = self.HID_RX_SIZE
                while this_len == self.HID_RX_SIZE and (real_bytes_rcvd < n or n != 1):

                    real_bytes = self._hid_read_singleshot(RX_TIMEOUT)
                    self._hid_buffer += real_bytes  # remove garbage after recording the response length
                    real_bytes_rcvd += len(real_bytes)
                    this_len = self.HID_RX_SIZE if len(
                        real_bytes) > 0 else 0  # len(d)

            if n == -1 or n % self.HID_RX_SIZE == 0:
                n = len(self._hid_buffer)
            if len(self._hid_buffer) > n:
                # take partial/whole buffer
                reply = self._hid_buffer[0:n]
                self._hid_buffer = self._hid_buffer[n:]
            else:
                # take what's available in buffer (timeout)
                reply = self._hid_buffer
                self._hid_buffer = b""
            # max(real_bytes_rcvd, len(self._hid_buffer)) # TODO: This will lead to one more read than needed, do better?
            self._in_waiting = len(self._hid_buffer)

            return reply

        if self._net != None:
            rem_n = n

            if rem_n == 0:
                return reply  # skip the rest
            if len(self._peek) > 0:
                if len(self._peek) > rem_n:
                    reply += self._peek[0:rem_n]
                    self._peek = self._peek[rem_n:]
                    rem_n = 0
                else:  # drain rest of the peek data
                    reply += self._peek
                    rem_n -= len(self._peek)
                    self._peek = b''
            if rem_n > 0:
                reply += self._request.raw.read(rem_n)

            if n == -1:
                n = len(reply)
            if len(reply) != n:
                Log.w(TAG, "Warning: More bytes returned than requested!")

            self._in_waiting -= n
            if len(self._peek) == 0:
                self._peek = self._request.raw.read(1)
                self._in_waiting = len(self._peek)

            if self._request.ok and self._com != None:  # clear serial on net success
                if self._serial.is_open and self._serial.in_waiting > 0:
                    self._serial.reset_input_buffer()
            return reply  # stop on success

        if self._com != None:
            if not self._serial.is_open:
                self._serial.open()
            if n == -1:
                n = self.in_waiting
            reply = self._serial.read(n)
            # Log.d(f"Got {len(reply)} byte(s) with {self.in_waiting} waiting: {reply}")
            return reply  # stop on success

        Log.e(TAG, "ERROR: No available device phy for read().")
        return reply

    def read_all(self):
        """\
        Read all bytes currently available in the buffer of the OS.
        """
        return self.read(self.in_waiting)

    def read_until(self, expected='\n', size=None):
        """\
        Read until an expected sequence is found (line feed by default), or the size
        is exceeded ### or until timeout occurs. ###
        """
        lenterm = len(expected)
        line = bytearray()
        # timeout = Timeout(self._timeout)
        if isinstance(expected, str):
            expected = expected.encode()  # convert to bytestr
        while True:
            c = self.read(1)
            if c:
                line += c
                if line[-lenterm:] == bytes(expected):
                    break
                if size is not None and len(line) >= size:
                    break
            else:
                break
            # if timeout.expired():
            #     break
        # Log.w(len(line), bytes(line))
        return bytes(line)

    def reset_input_buffer(self):
        if self._com != None:
            if self._serial.is_open and self._serial.in_waiting > 0:
                self._serial.reset_input_buffer()
        if self._net != None:
            tossed = len(self._request.text)  # calling 'text' clears 'raw'
            self._in_waiting = 0
        if self._hid != None:
            self._hid_buffer = b""
            self._in_waiting = 0

    def reset_output_buffer(self):
        if self._com != None:
            if self._serial.is_open:
                self._serial.reset_output_buffer()
        if self._net != None:
            pass  # nothing to do here
        if self._hid != None:
            pass  # nothing to do here

    def _hid_read_singleshot(self, timeout=None):
        reply = b''
        try:
            if self._device != None and self._device._Device__dev != None:
                if timeout == None:
                    # 15 seconds max wait for a reply (or else app hangs forever)
                    timeout = 15000
                d = self._device.read(self.HID_RX_SIZE, timeout)
                # remove garbage after recording the response length
                reply += d.replace(b'\x00', b'')
                if d != b'' and d.replace(b'\x00', b'') == b'':
                    reply += b'\x00'
        except Exception as e:
            self.close()
            # Log.e("HID read error: " + str(e))
            raise sp.SerialException(e)
        return reply

    @property
    def port(self):
        return self._port

    @port.setter
    def port(self, value):
        self._port = value
        if self._port == None:
            self._com = None
            self._net = None
            self._hid = None
        elif ':' in self._port:
            self._com = None
            self._net = None
            self._hid = self._port
        elif ';' in self._port:
            self._com = self._port.split(';')[0]
            self._net = self._port.split(';')[1]
            self._hid = None
        elif self._port.count('.') == 3:
            self._com = None
            self._net = self._port
            self._hid = None
        else:
            self._com = self._port
            self._net = None
            self._hid = None
        if self._com != None:
            self._serial.port = self._com
        else:
            self._serial.close()
            self._serial.port = None

    @property
    def com_port(self):
        return self._com

    @com_port.setter
    def com_port(self, value):
        if value != None and self._net != None:
            self.port = "{};{}".format(value, self._net)
        elif value != None:
            self.port = value
        elif self._net != None:
            self.port = self._net
        else:
            self.port = None

    @property
    def net_port(self):
        return self._net

    @net_port.setter
    def net_port(self, value):
        if value != None and self._com != None:
            self.port = "{};{}".format(self._com, value)
        elif value != None:
            self.port = value
        elif self._com != None:
            self.port = self._com
        else:
            self.port = None

    @property
    def hid_port(self):
        return self._hid

    @hid_port.setter
    def hid_port(self, value):
        if value != None:
            self.port = value
        else:
            self.port = None

    @property
    def timeout(self):
        return self._timeout

    @timeout.setter
    def timeout(self, value):
        self._serial.timeout = value
        self._timeout = value

    @property
    def parity(self):
        return self._parity

    @parity.setter
    def parity(self, value):
        self._serial.parity = value
        self._parity = value

    @property
    def baudrate(self):
        return self._baudrate

    @baudrate.setter
    def baudrate(self, value):
        self._serial.baudrate = value
        self._baudrate = value

    @property
    def bytesize(self):
        return self._bytesize

    @bytesize.setter
    def bytesize(self, value):
        self._serial.bytesize = value
        self._bytesize = value

    @property
    def stopbits(self):
        return self._stopbits

    @stopbits.setter
    def stopbits(self, value):
        self._serial.stopbits = value
        self._stopbits = value

    @property
    def xonxoff(self):
        return self._xonxoff

    @xonxoff.setter
    def xonxoff(self, value):
        self._serial.xonxoff = value
        self._xonxoff = value

    @property
    def rtscts(self):
        return self._rtscts

    @rtscts.setter
    def rtscts(self, value):
        self._serial.rtscts = value
        self._rtscts = value

    @property
    def dsrdtr(self):
        return self._dsrdtr

    @dsrdtr.setter
    def dsrdtr(self, value):
        self._serial.dsrdtr = value
        self._dsrdtr = value

    @property
    def write_timeout(self):
        return self._write_timeout

    @write_timeout.setter
    def write_timeout(self, value):
        self._serial.write_timeout = value
        self._write_timeout = value

    @property
    def inter_byte_timeout(self):
        return self._inter_byte_timeout

    @inter_byte_timeout.setter
    def inter_byte_timeout(self, value):
        self._serial.inter_byte_timeout = value
        self._inter_byte_timeout = value

    @property  # read-only
    def is_open(self):
        self._is_open = False
        if self._net != None:
            self._is_open |= self._request.ok
        if self._com != None:
            self._is_open |= self._serial.is_open
        if self._hid != None:
            if self._device != None:
                self._is_open |= (self._device._Device__dev != None)
            # else, port definitely not open: '_is_open' already false
        return self._is_open

    @property  # read-only
    def in_waiting(self):
        if self._hid != None:
            # self.HID_RX_INTERVAL if len(self._hid_buffer) == 0 else 0
            RX_TIMEOUT = 0
            self._hid_buffer += self._hid_read_singleshot(RX_TIMEOUT)
            self._in_waiting = len(self._hid_buffer)
            # Log.d(self._in_waiting) # , self._hid_buffer)
        elif self._net != None:
            pass  # already set by call to 'write'
        else:
            self._in_waiting = self._serial.in_waiting
        return self._in_waiting
