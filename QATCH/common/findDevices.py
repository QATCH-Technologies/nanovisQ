"""
findDevices.py

Device discovery utilities for QATCH hardware on local networks.

This module provides the :class:`Discovery` class, which locates QATCH devices
connected via Ethernet (APIPA or local-network subnets) by combining ARP table
inspection, ICMP ping scanning, UDP time-synchronisation broadcasts, and HTTP
endpoint queries.  It supports Windows, macOS, and Linux hosts.

Author(s):
    Alexander J. Ross (alexander.ross@qatchtech.com)
    Paul MacNichol  (paul.macnichol@qatchtech.com)
    Other QATCH Technologies contributors

Date:
    2026-03-20
"""

import platform
from socket import AF_INET, SO_BROADCAST, SO_REUSEADDR, SOCK_DGRAM, SOL_SOCKET, socket
import subprocess
from subprocess import PIPE, Popen
import sys
from threading import Thread

from progressbar import Bar, Percentage, ProgressBar, Timer
import requests

from QATCH.common.architecture import Architecture, OSType
from QATCH.common.logger import Logger as Log

TAG = "[Discovery]"

###############################################################################
# Handles checking and updating an QATCH device firmware on Teensy 3.6 boards
###############################################################################


class Discovery:
    """Discovers QATCH hardware devices on the local network.

    Uses a combination of ARP table inspection, UDP broadcast time-sync, ICMP
    ping sweeps, and HTTP endpoint queries to enumerate connected QATCH devices.
    Supports Windows, macOS, and Linux.
    """

    def __init__(self, argv=sys.argv):
        """Initialises the Discovery instance.

        Args:
            argv: Command-line arguments passed to the process.  Defaults to
                ``sys.argv``.  Reserved for future use.
        """
        # Log.d(self)
        # Log.d(argv)
        pass

    def run(self):
        """Entry point when the module is executed directly.

        Invokes :meth:`do_discover` with default arguments and discards the
        return value.  Intended for quick command-line testing.
        """
        self.do_discover()

    def ping(self, host):
        """Sends a single ICMP ping and reports whether the host responded.

        Note that some hosts block ICMP echo requests at the firewall level and
        will therefore return ``False`` even when the IP is reachable.

        Args:
            host (str): IP address or hostname to ping.

        Returns:
            bool: ``True`` if the host replied within the timeout window,
            ``False`` otherwise.
        """
        # Option for the number of packets as a function of
        count = "-n" if platform.system().lower() == "windows" else "-c"
        timeout = "-w" if platform.system().lower() == "windows" else "-W"
        time = "250" if platform.system().lower() == "windows" else "1"
        # Building the command. Ex: "ping -n 1 -w 1 google.com"
        command = ["ping", count, "1", timeout, time, host]
        return subprocess.call(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT) == 0

    def _send_udp_timesync(self):
        """Broadcasts a UDP NTP-style datagram to wake APIPA-addressed devices.

        Sends a single UDP packet to the link-local broadcast address
        ``169.254.255.255`` on port 123 (NTP).  This triggers QATCH devices
        that are waiting for a time-sync signal before they respond to ARP
        queries, making them visible to subsequent subnet scans.
        """
        cs = socket(AF_INET, SOCK_DGRAM)
        cs.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
        cs.setsockopt(SOL_SOCKET, SO_BROADCAST, 1)
        # Line broken up to fix E501 / B950
        datagram = (
            b"\x1c\x01\r\xe3\x00\x00\x00\x10\x00\x00\x00 NIST\xe6\xb2.\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\xe6\xb2.n@\xeb\x81\x7f\xe6\xb2.n@\xeb\x9c\xe8"
        )
        cs.sendto(datagram, ("169.254.255.255", 123))

    def _get_arp_subnets(self):
        """Extracts scannable subnets from the system's ARP cache.

        On Windows, parses ``arp -a`` output to find interface IP addresses and
        derives ``/24`` subnet prefixes for APIPA ranges (``169.254.x.x``) and,
        when ``scan_local_network`` is enabled, for other local subnets.  On
        non-Windows platforms the ARP output is pre-filtered by a MAC-address
        prefix specific to QATCH hardware.

        Returns:
            list[str]: Subnet strings with the host octet replaced by ``"x"``
            (e.g. ``"169.254.73.x"``), ready for IP substitution during
            scanning.
        """
        scan_local_network = False
        if Architecture.get_os() is OSType.windows:
            arp_task = Popen(["arp", "-a"], shell=True, stdout=PIPE, stderr=PIPE)
        else:
            arp_task = Popen(['arp -a | grep " 4:e9:e5:"'], shell=True, stdout=PIPE, stderr=PIPE)

        output = arp_task.communicate()[0].decode("utf-8").split("\n")
        subnets = []

        for line in output:
            if not line.strip():
                continue

            if Architecture.get_os() is OSType.windows:
                if "Interface" in line:
                    ip = line.strip().split(" ")[1]
                    if ip.startswith("169.254"):
                        subnets.append("169.254.73.x")
                    elif scan_local_network:
                        ip_parts = ip.split(".")
                        subnets.append(f"{ip_parts[0]}.{ip_parts[1]}.{ip_parts[2]}.x")
            else:
                ip = line.split(" ")[1][1:-1]
                subnets.append(ip)
        return subnets

    def scan_subnets(self):
        """Performs a threaded ICMP ping sweep across all discovered subnets.

        First broadcasts a UDP time-sync packet, then retrieves the list of
        target subnets from the ARP table.  Up to ``max_threads`` concurrent
        threads are used to ping each host in the ``1-254`` range of every
        subnet.  Progress is displayed via a :class:`ProgressBar`.

        Returns:
            tuple[int, list[str]]: A 2-tuple of ``(count, found)`` where
            *count* is the number of responsive hosts and *found* is a sorted
            list of their IP address strings.
        """
        self._send_udp_timesync()
        subnets = self._get_arp_subnets()

        if not subnets:
            Log.e("ERROR: APIPA subnet not available. Re-power device(s) and check connections.")

        bar = ProgressBar(
            widgets=[TAG, " ", Bar(marker=">"), " ", Percentage(), " ", Timer()]
        ).start()

        max_threads = 25
        active = []
        found = []

        def scan(ip):
            try:
                active.append(ip)
                if self.ping(ip):
                    found.append(ip)
            except Exception as e:
                Log.e(f"Thread error for {ip}: {e}")
            finally:
                active.remove(ip)

        bar.maxval = len(subnets) * 254
        i = 0
        for sn in subnets:
            for c in range(1, 255):
                while len(active) >= max_threads:
                    pass
                ip = sn.replace("x", str(c))
                thread = Thread(target=scan, args=(ip,))
                thread.start()
                i += 1
                bar.update(i)

        while active:
            pass
        bar.finish()
        found.sort()
        return len(found), found

    def _get_arp_devices(self):
        """Returns IP addresses of QATCH devices found in the ARP cache.

        Filters ``arp -a`` output by the QATCH-specific MAC prefix
        ``04-e9-e5-`` (Windows) or ``4:e9:e5:`` (POSIX) to identify only
        known QATCH hardware rather than every host on the network.

        Returns:
            list[str]: IP address strings of matching ARP entries.
        """
        if Architecture.get_os() is OSType.windows:
            arp_task = Popen(["arp", "-a"], shell=True, stdout=PIPE, stderr=PIPE)
        else:
            arp_task = Popen(['arp -a | grep " 4:e9:e5:"'], shell=True, stdout=PIPE, stderr=PIPE)

        output = arp_task.communicate()[0].decode("utf-8").split("\n")
        devices = []
        for line in output:
            if not line.strip():
                continue

            if Architecture.get_os() is OSType.windows:
                if "04-e9-e5-" in line:
                    devices.append(line.strip().split(" ")[0])
            else:
                devices.append(line.split(" ")[1][1:-1])
        return devices

    def _fetch_endpoint_with_retry(self, url, retries=3):
        """Fetches a device HTTP endpoint, retrying until a non-empty response is received.

        Args:
            url (str): Full URL of the device endpoint to query.
            retries (int): Maximum number of attempts before giving up.
                Defaults to ``3``.

        Returns:
            list[str]: Lines of the decoded response body, or an empty list if
            all attempts returned a blank response or raised an exception.
        """
        for _ in range(retries):
            response = requests.get(url, timeout=1)
            response.raise_for_status()
            if len(response.content) > 0:
                return response.content.decode("utf-8").split("\n")
        Log.e(f"ERROR: HTTP response from `{url}` was blank!")
        return []

    def _parse_device_info(self, content):
        """Parses colon-delimited key-value lines from the ``/info`` endpoint.

        Args:
            content (list[str]): Lines returned by the device's ``/info``
                HTTP endpoint.

        Returns:
            dict[str, str]: Mapping of field names to their values (e.g.
            ``{"HW": "v2", "IP": "169.254.73.42", ...}``).  Lines without a
            colon separator are silently ignored.
        """
        data = {}
        for line in content:
            if ":" in line:
                k, v = line.split(":", 1)
                data[k.strip()] = v.strip()
        return data

    def _parse_device_version(self, content):
        """Extracts build identifier, version string, and build date from the ``/version`` endpoint.

        Expects the response to contain at least three lines in the order:
        ``build``, ``version``, ``date``.

        Args:
            content (list[str]): Lines returned by the device's ``/version``
                HTTP endpoint.

        Returns:
            tuple[str, str, str]: A 3-tuple of ``(build, version, date)``
            strings.  Returns three empty strings if *content* contains fewer
            than three lines.
        """
        if len(content) >= 3:
            return content[0].strip(), content[1].strip(), content[2].strip()
        return "", "", ""

    def _query_device(self, dev):
        """Queries a device's HTTP endpoints and validates its reported IP address.

        Fetches both the ``/info`` and ``/version`` endpoints for the given
        device IP.  If the IP reported by the device matches *dev*, the method
        logs the discovery and returns the collected metadata.

        Args:
            dev (str): IP address of the device to query.

        Returns:
            list | None: An 8-element list
            ``[build, version, date, hw, ip, mac, usb, uid]`` on success, or
            ``None`` if the endpoints are unreachable, return empty content, or
            report a mismatched IP address.
        """
        try:
            info_content = self._fetch_endpoint_with_retry(f"http://{dev}:8080/info")
            version_content = self._fetch_endpoint_with_retry(f"http://{dev}:8080/version")

            if not info_content or not version_content:
                return None

            # Parse extracted content using helpers to reduce complexity
            info = self._parse_device_info(info_content)
            build, version, date = self._parse_device_version(version_content)

            hw = info.get("HW", "")
            ip = info.get("IP", "")
            mac = info.get("MAC", "")
            usb = info.get("USB", "")
            uid = info.get("UID", "")

            if dev == ip:
                Log.i(TAG, f"FOUND `{usb}` @ http://{ip}/")
                return [build, version, date, hw, ip, mac, usb, uid]

            Log.e(TAG, f"IP Mismatch! (IP: {dev} != {ip})")

        except requests.exceptions.RequestException as e:
            Log.e(TAG, f"HTTP request exception for device `{dev}`: {type(e).__name__}")
        except Exception as e:
            Log.e(TAG, f"An unexpected exception occurred with device `{dev}`: {e}")

        return None

    def do_discover(self, full_query=False):
        """Discovers QATCH devices on the network, optionally querying each one.

        When *full_query* is ``False`` (the default), returns the raw list of
        IP addresses found in the ARP cache without making any HTTP requests.
        When *full_query* is ``True``, each candidate IP is queried via HTTP
        and only devices that respond with valid metadata are returned.

        Args:
            full_query (bool): If ``True``, query each discovered device's
                HTTP endpoints and return structured metadata.  Defaults to
                ``False``.

        Returns:
            list: If *full_query* is ``False``, a list of IP address strings.
            If *full_query* is ``True``, a list of 8-element lists
            ``[build, version, date, hw, ip, mac, usb, uid]``, one per
            successfully queried device.
        """
        devices = self._get_arp_devices()

        if not full_query:
            return devices

        if not devices:
            Log.d(
                TAG, "No QATCH hardware devices found on the network. Please check connection(s)."
            )
            return []

        enumerated = []
        for dev in devices:
            result = self._query_device(dev)
            if result:
                enumerated.append(result)

        return enumerated


if __name__ == "__main__":
    Discovery().run()
