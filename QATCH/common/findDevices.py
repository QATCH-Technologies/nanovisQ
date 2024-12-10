from datetime import datetime
from subprocess import Popen, PIPE
from time import time, sleep
import sys
import requests
from threading import Thread
import subprocess
import platform
from socket import *

from progressbar import Bar, Percentage, ProgressBar, RotatingMarker, Timer
from QATCH.common.architecture import Architecture, OSType
from QATCH.common.logger import Logger as Log


TAG = "[Discovery]"

###############################################################################
# Handles checking and updating an QATCH device firmware on Teensy 3.6 boards
###############################################################################


class Discovery:

    def __init__(self, argv=sys.argv):
        # Log.d(self)
        # Log.d(argv)
        pass

    def run(self):
        self.doDiscover()

    def ping(self, host):
        """
        Returns True if host (str) responds to a ping request.
        Remember that a host may not respond to a ping (ICMP) request even if the host name is valid.
        """
        # Option for the number of packets as a function of
        count = '-n' if platform.system().lower() == 'windows' else '-c'
        timeout = '-w' if platform.system().lower() == 'windows' else '-W'
        time = '250' if platform.system().lower() == 'windows' else '1'
        # Building the command. Ex: "ping -n 1 -w 1 google.com"
        command = ['ping', count, '1', timeout, time, host]
        return subprocess.call(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT) == 0

    def scanSubnets(self):
        scan_local_network = False

        # attempt UDP timesync for APIPA devices, broadcast only, no listen (to avoid firewall issues)
        cs = socket(AF_INET, SOCK_DGRAM)
        cs.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
        cs.setsockopt(SOL_SOCKET, SO_BROADCAST, 1)
        datagram = b'\x1c\x01\r\xe3\x00\x00\x00\x10\x00\x00\x00 NIST\xe6\xb2.\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xe6\xb2.n@\xeb\x81\x7f\xe6\xb2.n@\xeb\x9c\xe8'
        # contains dummy timestamp from 8/25/2022, but this is just to sync LEDs; precision is important, accuracy is irrelevant
        cs.sendto(datagram, ('169.254.255.255', 123))

        if Architecture.get_os() is OSType.windows:
            arp_task = Popen(['arp', '-a'],
                             shell=True, stdout=PIPE, stderr=PIPE)
        else:
            arp_task = Popen(['arp -a | grep " 4:e9:e5:"'],
                             shell=True, stdout=PIPE, stderr=PIPE)

        output = arp_task.communicate()[0]
        output = output.decode("utf-8").split("\n")

        # Initialize ProgressBar
        bar = ProgressBar(widgets=[TAG, ' ', Bar(
            marker='>'), ' ', Percentage(), ' ', Timer()]).start()
        bar.maxval = 256  # assume one whole subnet will be scanned, update later once we know more
        i = 0
        bar.update(i)
        subnets = []
        enumerated = []
        for line in output:
            if not line.strip():
                continue  # ignore blank lines

            if Architecture.get_os() is OSType.windows:
                if 'Interface' in line:
                    ip = line.strip().split(" ")[1]
                    ip_parts = ip.split(".")
                    if ip.startswith('169.254'):
                        ip = f"169.254.73.x"
                    else:
                        if not scan_local_network:
                            continue  # only scan APIPA network subnet 73
                        ip = f"{ip_parts[0]}.{ip_parts[1]}.{ip_parts[2]}.x"
                    subnets.append(ip)
                    i += 1
                    bar.update(i)  # increment by 1
            else:
                ip = line.split(" ")[1][1:-1]
                subnets.append(ip)

        if len(subnets) == 0:
            Log.e(
                "ERROR: APIPA subnet not available. Re-power device(s) and check connections.")

        # Log.d(f"Available subnets: {subnets}")
        max_threads = 25
        active = []
        found = []

        def ping(host):
            """
            Returns True if host (str) responds to a ping request.
            Remember that a host may not respond to a ping (ICMP) request even if the host name is valid.
            """
            # Option for the number of packets as a function of
            count = '-n' if platform.system().lower() == 'windows' else '-c'
            timeout = '-w' if platform.system().lower() == 'windows' else '-W'
            time = '250' if platform.system().lower() == 'windows' else '1'
            # Building the command. Ex: "ping -n 1 -w 1 google.com"
            command = ['ping', count, '1', timeout, time, host]
            return subprocess.call(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT) == 0

        def scan(ip):
            try:
                # Log.d(f"Start {c}")
                active.append(ip)
                if ping(ip):
                    # Log.d(f"Found {ip}")
                    found.append(ip)
            except:
                Log.e(f"Thread error for {ip}")
            finally:
                active.remove(ip)

        # total number of IPs to be pinged
        bar.maxval = len(subnets) * (len(range(1, 255)) + 1)
        for sn in subnets:
            pool = range(1, 255)
            for c in pool:
                while (len(active) >= max_threads):
                    pass
                ip = sn.replace('x', str(c))
                thread = Thread(target=scan, args=(ip,))
                thread.start()
                i += 1
                bar.update(i)  # increment by 1
        while (len(active) > 0):
            pass
        bar.finish()
        found.sort()
        # Log.d(f"IPs: {found}")
        return len(found), found  # identify(found)

    ###########################################################################
    # Updates the running Teensy 3.6 device firmware to the Recommended version
    ###########################################################################

    def doDiscover(self, full_query=False):
        """
        :param port: Serial port name :type port: str.
        """
        # Check to make sure the teensy loader CLI (or other conflicts) are not Running

        # Command is based on OS running
        if Architecture.get_os() is OSType.windows:
            arp_task = Popen(['arp', '-a'],
                             shell=True, stdout=PIPE, stderr=PIPE)
        else:
            arp_task = Popen(['arp -a | grep " 4:e9:e5:"'],
                             shell=True, stdout=PIPE, stderr=PIPE)

        output = arp_task.communicate()[0]
        output = output.decode("utf-8").split("\n")

        devices = []
        enumerated = []
        for line in output:
            if not line.strip():
                continue  # ignore blank lines

            if Architecture.get_os() is OSType.windows:
                if '04-e9-e5-' in line:
                    ip = line.strip().split(" ")[0]
                    devices.append(ip)
            else:
                ip = line.split(" ")[1][1:-1]
                devices.append(ip)

        if not full_query:
            return devices
        if not len(devices) > 0:
            Log.d(
                "No QATCH hardware devices found on the network. Please check connection(s) and try again.")

        for dev in devices:
            try:
                for i in range(3):
                    info_response = requests.get(
                        'http://' + dev + ':8080/info', timeout=1)
                    info_response.raise_for_status()

                    # skip 3x retry loop if content rx'd
                    if len(info_response.content) > 0:
                        break
                    Log.e(
                        "ERROR: HTTP response from device '{}' was blank!".format(dev))

                for i in range(3):
                    version_response = requests.get(
                        'http://' + dev + ':8080/version', timeout=1)
                    version_response.raise_for_status()

                    # skip 3x retry loop if content rx'd
                    if len(version_response.content) > 0:
                        break
                    Log.e(
                        "ERROR: HTTP response from device '{}' was blank!".format(dev))

                content = info_response.content.decode("utf-8").split("\n")
                hw = ip = mac = usb = uid = ""
                for line in content:
                    line = line.split(":", 1)
                    if line[0] == "HW":
                        hw = line[1].strip()
                    if line[0] == "IP":
                        ip = line[1].strip()
                    if line[0] == "MAC":
                        mac = line[1].strip()
                    if line[0] == "USB":
                        usb = line[1].strip()
                    if line[0] == "UID":
                        uid = line[1].strip()

                content = version_response.content.decode("utf-8").split("\n")
                build = version = date = ""
                for i in range(len(content)):
                    if i == 0:
                        build = content[i].strip()
                    if i == 1:
                        version = content[i].strip()
                    if i == 2:
                        date = content[i].strip()

                if dev == ip:
                    Log.i("FOUND '{}' @ http://{}/".format(usb, ip))
                    enumerated.append(
                        [build, version, date, hw, ip, mac, usb, uid])
                else:
                    Log.e("Error: IP Mismatch! (IP: {} != {})".format(dev, ip))

            except requests.exceptions.ConnectionError:
                Log.e("ERROR: Unable to connect to device '{}'!".format(dev))
            except requests.exceptions.HTTPError:
                Log.e(
                    "ERROR: HTTP request to device '{}' was not successful!".format(dev))
            except requests.exceptions.Timeout:
                Log.e(
                    "ERROR: HTTP request to device '{}' had a timeout occur!".format(dev))
            except requests.exceptions.TooManyRedirects:
                Log.e(
                    "ERROR: HTTP request to device '{}' had too many redirects!".format(dev))
            except requests.exceptions.RequestException:
                Log.e(
                    "ERROR: HTTP request to device '{}' experienced an exception!".format(dev))
            except:
                Log.e(
                    "ERROR: An unexpected exception occurred communicating with device '{}'!".format(dev))

        return enumerated


if __name__ == '__main__':
    Discovery().run()
