/**************************************************************************/
/*!
    @file     NVMEM.h
    @author   Alexander J. Ross (QATCH Technologies LLC)

  @section  HISTORY

    v1.0  - First release
*/
/**************************************************************************/

#ifndef NVMEM_H
#define NVMEM_H

#include "Arduino.h"
#include "EEPROM.h"

// #define NVMEM_DEBUG

#define NVMEM_LENGTH EEPROM.length()
#define NVMEM_INVERT true
#define NVMEM_OFFSET 0

// HW Revisions supported:
#define HW_REVISION_X 0xFF // Rev 0xFF = attempt auto-detection, set if found (default)
#define HW_REVISION_0 0    // Rev 0 = no LCD; MAX config A; blue LED support; 125MHz xtal
#define HW_REVISION_1 1    // Rev 1 = w/ LCD; MAX config B; multi LED support; 125MHz xtal
#define HW_REVISION_2 2    // Rev 2 = w/ LCD; MAX config B; multi LED support; 30MHz xtal (6x multiplier enabled to 180MHz)
/// @note Add new HW revisions here...

/// @note Increment NVMEM_VERSION each time you change NvMem_RAM structure!
#define NVMEM_VERSION 2

struct NvMem_RAM
{
  byte version;     // -1: read only, expect for in update() method
  byte pid;         // 0: port ID, for multiplexed devices
  byte OffsetA;     // 1: always (FW fixed offset, in EEPROM)
  byte HW_Revision; // 2: see 'HW_REVISION_[#]' #defines for supported values
  byte OffsetM;     // 3: measuring (FW fixed offset, in EEPROM)
  byte Ethernet_EN; // 4: enable Ethernet PHY to report an IP (if chip present) - default: disable
  // byte NewValue;
  /// @note Add new entries here, even if inverted!
  /// @note Also increment NVMEM_VERSION, add a value in defaults(), and add logic in update()
} __attribute__((packed));

class NvMem
{
public:
  NvMem();
  NvMem_RAM mem;            // get and set from this struct, be sure to call load() and/or defaults() first!
  NvMem_RAM defaults(void); // returns entire struct with defaults, must be set and saved to be permanent; calling makes 'mem' valid
  bool erase(void);         // clears NVMEM to 0xFFs
  bool load(void);          // returns False if entire EEPROM region is all 0xFFs, in which case, defaults() must be called, set and saved
  bool save(void);          // store to EEPROM, only if struct_is_valid
  byte update(void);        // inits struct objects with uninit'd values (all b1's), returns number of defaulted objects (if not zero, must save)
  bool isValid(void);       // indicates whether the 'nv.mem' structure is valid (i.e. has been initialized with non-emtpy EEPROM data)
  const int size = sizeof(NvMem_RAM) / sizeof(byte);

private:
  byte *serialize(byte arr[], NvMem_RAM data);  // convert the given structure into a serialized byte array (passsed by reference)
  NvMem_RAM deserialize(byte *arr, size_t len); // return the given serialized byte array as an 'NvMem_RAM' structure of given length
  void reverse(byte *arr, size_t len);          // reverse the values in a given array (passed by reference)
  bool struct_is_valid = false;                 // disallow saving or using an invalid struct, see 'isValid()' for public access
};

#endif
