//******************************************************************************
// FlasherX -- firmware "OTA" update via Intel Hex file over serial byte stream
//******************************************************************************
//
// WARNING: You can brick your Teensy with incorrect flash erase/write, such as
// incorrect flash config (0x400-40F). This code may or may not prevent that.
//
// Based on Flasher3 (Teensy 3.x) and Flasher4 (Teensy 4.x) by Jon Zeeff
//
// Jon Zeeff 2016, 2019, 2020
// This code is in the public domain.  Please retain my name and
// in distributed copies, and let me know about any bugs
//
// I, Jon Zeeff, give no warranty, expressed or implied for this software
// and/or documentation provided, including, without limitation, warranty of
// merchantability and fitness for a particular purpose.

// 7th Modifications 01/07/22 updates by Joe Pasquariello (v2.1)
//   - FlashTxx.cpp delete local T4x wait/write/erase to use functions in TD 1.56
//   - FlashTxx.h update FLASH_SIZE for Teensy Micromod from 8 to 16 MB
//   - FlasherX.ino update to print "FlasherX OTA firmware update v2.1" on bootup
//   - add option to artificially increase code size via const array (in flash)
// 6th Modifications 11/18/21 bug fix in file FlashTXX.cpp by Joe Pasquariello
//   - fix logic in while loop in flash_block_write() in FlashTXX
// 5th Modifications 10/27/21 add support for Teensy Micromod by Joe Pasquariello
//   - define macros for TEENSY_MICROMOD w/ same values as for TEENSY40
//   - update FLASH_SIZE for T4.1 and TMM from 2MB to 8MB
// 4th Modifications merge of Flasher3/4 and new features by Joe Pasquariello
//   - FLASH buffer dynamically sized from top of existing code to FLASH_RESERVE
//   - optional RAM buffer option for T4.x via macro RAM_BUFFER_SIZE > 0
//   - Stream* (USB or UART) and buffer addr/size set at run-time
//   - incorporate Frank Boesing's FlashKinetis routines for T3.x
//   - add support for Teensy 4.1 and Teensy LC
//    This code is released into the public domain.
// 3rd Modifications for T3.5 and T3.6 in Dec 2020 by Joe Pasquariello
//    This code is released into the public domain.
// 2nd Modifications for teensy 3.5/3/6 by Deb Hollenback at GiftCoder
//    This code is released into the public domain.
//    see https://forum.pjrc.com/threads/43165-Over-the-Air-firmware-updates-changes-for-flashing-Teensy-3-5-amp-3-6
// 1st Modifications by Jon Zeeff
//    see https://forum.pjrc.com/threads/29607-Over-the-air-updates
// Original by Niels A. Moseley, 2015.
//    This code is released into the public domain.
//    https://namoseley.wordpress.com/2015/02/04/freescale-kinetis-mk20dx-series-flash-erasing/

#include <Arduino.h>
#include "FlashTxx.h"		// TLC/T3x/T4x/TMM flash primitives

#if defined(__IMXRT1062__) && defined(ARDUINO_TEENSY41)
  #include <NativeEthernet.h>
#endif

#include <stdio.h>    // sscanf(), etc.
#include <string.h>   // strlen(), etc.

//******************************************************************************
// hex_info_t	struct for hex record and hex file info
//******************************************************************************
typedef struct {	// 
  char *data;		// pointer to array allocated elsewhere
  unsigned int addr;	// address in intel hex record
  unsigned int code;	// intel hex record type (0=data, etc.)
  unsigned int num;	// number of data bytes in intel hex record
 
  uint32_t base;	// base address to be added to intel hex 16-bit addr
  uint32_t min;		// min address in hex file
  uint32_t max;		// max address in hex file
  
  int eof;		// set true on intel hex EOF (code = 1)
  int lines;		// number of hex records received  
} hex_info_t;

void read_ascii_line( Stream *serial, char *line, int maxbytes );
int  parse_hex_line( const char *theline, char *bytes,
	unsigned int *addr, unsigned int *num, unsigned int *code );
int  process_hex_record( hex_info_t *hex );
void update_firmware( Stream *serial, uint32_t buffer_addr, uint32_t buffer_size );

void flasher_update_task( Stream *client, uint8_t ledPin=13 );
