# QATCH

## Name
QATCH Technologies Q-1 Device Firmware

## Programming language
Teensyduino / C++

## Required Libraries
- ADC v9.0 https://github.com/pedvide/ADC
- TeensyID v1.3.1 https://github.com/sstaub/TeensyID

## Project Settings

***For Teensy 3.6 hardware***

	Board: "Teensy 3.6"
	USB Type: "Serial"
	CPU Speed: "180 MHz"
	Optimize: "Fastest + pure-code with LTO"

***For Teensy 4.1 hardware***

	Board: "Teensy 4.1"
	USB Type: "Serial"
	CPU Speed: "600 MHz"
	Optimize: "Fastest"

NOTE: You MUST manually set these settings in Teensyduino under the "Tools" menu to reproduce the build output for this project.
The default project compiler settings will yield a slower program output than desired.

## License and Citations
The project is distributed under GNU GPLv3 (General Public License).
