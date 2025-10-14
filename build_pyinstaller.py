import subprocess
import logging
import sys
import threading
from colorama import init, Fore, Style
from pathlib import Path
from dateutil import parser

# Files
WARNING_LOG_FILE = "build_warnings.txt"
MAIN_LOG_FILE = "build_log.txt"

init(autoreset=True)

# all lists contain leader ("time(ms) levelname:")
# leader must be removed before comparing values
# using `set()` auto sorts, which we do not want
previous_warnings = []
seen_warnings = []
new_warnings = []
missing_warnings = []
merged_warnings = []

runtime = 0
levelmap = logging.getLevelNamesMapping()


# Load previous warnings
if Path(WARNING_LOG_FILE).exists():
    with open(WARNING_LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            previous_warnings.append(line)


def is_timestamp(string: str) -> bool:
    try:
        parser.parse(string)
        return True
    except (ValueError, TypeError):
        return False


def strip_time_and_levelname(line: str) -> str:
    if line.find(":"):
        # parse timestamp (in ms) and log level from line (if present)
        start_of_line = line.strip().split(":")[0]
        if start_of_line.count(" ") == 1:
            split_of_line = start_of_line.split()
            if split_of_line[0].isnumeric():
                level = levelmap.get(split_of_line[1], logging.NOTSET)
                if level > logging.NOTSET:
                    # take off leader time/level
                    line = line.split(":", maxsplit=1)[-1]
            elif is_timestamp(line.split()[1]):
                # take off "W[xxxx] [timestamp] [runtime]"
                line = line.split(maxsplit=3)[-1]
    return line.strip()

# Formatter with color for new warnings


class SmartFormatter(logging.Formatter):
    def format(self, record):
        line = record.getMessage()
        msg = strip_time_and_levelname(line)
        color = ""
        if record.levelno == logging.WARNING:
            new = True
            for prev in previous_warnings:
                prev = strip_time_and_levelname(prev)
                if msg == prev:
                    new = False
                    break
            if new:
                color = Fore.RED                    # new warning!
                new_warnings.append(line)
            else:
                color = Fore.YELLOW + Style.BRIGHT  # old warning
        elif record.levelno == logging.ERROR:
            color = Fore.RED
        elif record.levelno == logging.INFO:
            color = Fore.CYAN
        else:  # DEBUG, NOTSET, unknown
            color = Fore.WHITE
        return f"{color}{super().format(record)}{Style.RESET_ALL}"


# Setup logger
logger = logging.getLogger("build_watchdog")
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(MAIN_LOG_FILE, mode='w', encoding='utf-8')
file_handler.setFormatter(logging.Formatter("%(message)s"))

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(SmartFormatter("%(message)s"))

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Capture subprocess output


def stream_output(pipe):
    level = logging.NOTSET
    for line in iter(pipe.readline, ''):
        line = str(line).strip()
        if line.find(":"):
            # parse timestamp (in ms) and log level from line (if present)
            start_of_line = line.split(":")[0]
            if start_of_line.count(" ") == 1:
                split_of_line = start_of_line.split()
                if split_of_line[0].isnumeric():
                    runtime = int(split_of_line[0])
                    level = levelmap.get(split_of_line[1], logging.ERROR)
        if not line:
            continue
        if level >= logging.WARNING:
            seen_warnings.append(line)
        logger.log(level, line)
    if level != logging.NOTSET:
        logger.debug(f"Total runtime: {runtime} ms")
    pipe.close()

# Run subprocess and capture


def run_and_log(command):
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        bufsize=1
    )
    threads = [
        threading.Thread(target=stream_output, args=(process.stdout,)),
        threading.Thread(target=stream_output, args=(process.stderr,))
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    process.wait()


# Main usage
run_and_log(['pyinstaller',
             '--log-level', logging.getLevelName(logging.INFO),
             'QATCH nanovisQ.spec'])

# Calculate missing warnings (if any)
all_same = len(previous_warnings) == len(seen_warnings) \
    and len(new_warnings) == 0
if all_same:
    logger.debug("No new warnings/errors encountered.")
else:
    merged_warnings.extend(previous_warnings)
    for warning in previous_warnings:
        prev = strip_time_and_levelname(warning)
        found = False
        for seen in seen_warnings:
            msg = strip_time_and_levelname(seen)
            if msg == prev:
                found = True
                break  # next prev
        if not found:
            missing_warnings.append(warning)

# Log new warnings at end (if any)
if len(new_warnings):
    logger.error("New warnings/errors encountered:")
    for warning in new_warnings:
        logger.error(warning)

# Log missing warnings at end (if any)
if len(missing_warnings):
    logger.error("Missing warnings/errors encountered:")
    for warning in missing_warnings:
        logger.error(warning)

# Ask what to do about new warnings (if any)
if len(new_warnings):
    if input(
            "Should new entries be added to the list of acceptable warnings (Y/N)? ").upper() == "Y":
        merged_warnings.extend(new_warnings)

# Ask what to do about missing warnings (if any)
if len(missing_warnings):
    if input(
            "Should missing entries be removed to the list of acceptable warnings (Y/N)? ").upper() == "Y":
        for missing in missing_warnings:
            if missing in merged_warnings:
                merged_warnings.remove(missing)

# Save current warnings for next run (if desired)
if len(merged_warnings):
    with open(WARNING_LOG_FILE, "w", encoding="utf-8") as f:
        for warning in merged_warnings:
            f.write(warning + "\n")
