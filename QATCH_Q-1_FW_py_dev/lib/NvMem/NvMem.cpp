/**************************************************************************/
/*!
    @file     NvMem.cpp
    @author   Alexander J. Ross (QATCH Technologies LLC)

  @section  HISTORY

    v1.0 - First release
*/
/**************************************************************************/

#include "NvMem.h"

NvMem::NvMem()
{
  struct_is_valid = false;
}

/**
 * @brief Construct the default NvMem_RAM structure and mark the in-memory NVM as valid.
 *
 * Returns an NvMem_RAM populated with the canonical defaults (version, pid, offsets,
 * hardware revision, Ethernet flag, and POGO position/move-delay defaults).
 *
 * The function also sets internal struct_is_valid to true so the defaults may be
 * saved back to EEPROM. Callers must either call save() followed by load() after
 * using these defaults, or avoid calling defaults() when they do not intend to
 * persist/refresh the stored NVM state.
 *
 * Note: EEPROM entries filled with 0xFF are treated as "uninitialized"; update()
 * relies on this convention to initialize new fields. Default values must be chosen
 * so that 0xFF either represents an invalid enumeration value (forcing a version bump)
 * or equals the desired default.
 *
 * @return NvMem_RAM The initialized defaults ready for serialization/persistence.
 */
NvMem_RAM NvMem::defaults(void)
{
  /// The all b1's value for any entry in EEPROM
  /// must satisfy one of the following conditions:
  ///  - The value is NOT VALID for the enumeration (must increment NVMEM_VERSION)
  ///  - The value is DEFAULT for the enumeration (optional increment NVMEM_VERSION)
  /// If not followed, update() cannot init new entries

  NvMem_RAM defaults;

  /// @note initialize defaults for all entries here:
  defaults.version = NVMEM_VERSION;
  defaults.pid = 0xFF;
  defaults.OffsetA = 0;
  defaults.HW_Revision = HW_REVISION_X;
  defaults.OffsetM = 0;
  defaults.Ethernet_EN = 0;
  defaults.POGO_PosOpened = DEFAULT_POS_OPENED;
  defaults.POGO_PosClosed = DEFAULT_POS_CLOSED;
  defaults.POGO_MoveDelay = DEFAULT_MOVE_DELAY;
  // defaults.NewValue = 42;

  struct_is_valid = true; // required for saving defaults, which is required to come soon after this call
  /// @attention Calling code must either call save() and load() after calling defaults() to refresh isValid()
  ///            -OR- caller must block call to defaults() if !isValid() and not planning to call load() again

  return defaults;
}

bool NvMem::erase(void)
{
  const int n = size;
  for (int i = 0; i < n; i++)
  {
#if NVMEM_INVERT
    int address = NVMEM_LENGTH - NVMEM_OFFSET - n + i;
#else
    int address = NVMEM_OFFSET + i;
#endif
    EEPROM.update(address, 0xFF);
#ifdef NVMEM_DEBUG
    Serial.print("Erasing EEPROM address ");
    Serial.print(address);
#endif
  }
#if NVMEM_INVERT
  for (int i = NVMEM_LENGTH - NVMEM_OFFSET - n - 1; i >= 0; i--)
#else
  for (int i = NVMEM_OFFSET + n; i <= NVMEM_LENGTH - 1; i++)
#endif
  {
    if (EEPROM.read(i) != 0xFF)
      EEPROM.write(i, 0xFF);
    else
    {
#ifdef NVMEM_DEBUG
      Serial.print("Stopped scanning for empty EEPROM at address ");
      Serial.println(i);
#endif
      break;
    }
  }
  return true;
}

/**
 * @brief Ensure stored NvMem_RAM fields are populated with sensible defaults.
 *
 * Compares the in-memory NvMem_RAM (mem) against the current defaults and updates
 * any fields that are unset (0xFF) when a valid default exists. Always updates
 * the version field if it differs from the default.
 *
 * Side effects:
 * - Mutates members of `mem` for each updated field.
 * - Increments an internal count for each modified field.
 *
 * @return byte Number of fields modified. If the NvMem structure is not valid,
 *               the function returns (byte)-1 (0xFF) to indicate failure.
 */
byte NvMem::update(void)
{
  if (!struct_is_valid)
  {
#ifdef NVMEM_DEBUG
    Serial.println("NVMEM struct is NOT VALID. Cannot update().");
#endif
    return -1;
  }

  byte modified_entries = 0;
  NvMem_RAM DEFAULT = defaults();

  if (mem.version != DEFAULT.version)
  {
    mem.version = DEFAULT.version;
    modified_entries++;
  }
  /// @todo instead of an ever growing list of if statement, make this into a for loop for each value in the struct
  if (mem.pid == 0xFF && DEFAULT.pid != 0xFF)
  {
    mem.pid = DEFAULT.pid;
    modified_entries++;
  }
  if (mem.OffsetA == 0xFF && DEFAULT.OffsetA != 0xFF)
  {
    mem.OffsetA = DEFAULT.OffsetA;
    modified_entries++;
  }
  if (mem.HW_Revision == 0xFF && DEFAULT.HW_Revision != 0xFF)
  {
    mem.HW_Revision = DEFAULT.HW_Revision;
    modified_entries++;
  }
  if (mem.OffsetM == 0xFF && DEFAULT.OffsetM != 0xFF)
  {
    mem.OffsetM = DEFAULT.OffsetM;
    modified_entries++;
  }
  if (mem.Ethernet_EN == 0xFF && DEFAULT.Ethernet_EN != 0xFF)
  {
    mem.Ethernet_EN = DEFAULT.Ethernet_EN;
    modified_entries++;
  }
  if (mem.POGO_PosOpened == 0xFF && DEFAULT.POGO_PosOpened != 0xFF)
  {
    mem.POGO_PosOpened = DEFAULT.POGO_PosOpened;
    modified_entries++;
  }
  if (mem.POGO_PosClosed == 0xFF && DEFAULT.POGO_PosClosed != 0xFF)
  {
    mem.POGO_PosClosed = DEFAULT.POGO_PosClosed;
    modified_entries++;
  }
  if (mem.POGO_MoveDelay == 0xFF && DEFAULT.POGO_MoveDelay != 0xFF)
  {
    mem.POGO_MoveDelay = DEFAULT.POGO_MoveDelay;
    modified_entries++;
  }
  // if (mem.NewValue == 0xFF && DEFAULT.NewValue != 0xFF)
  // {
  //   mem.NewValue = DEFAULT.NewValue;
  //   modified_entries++;
  // }

  /// @note Add update logic for new entries here!

#ifdef NVMEM_DEBUG
  Serial.print("NVMEM struct was out-of-date. Modified ");
  Serial.print(modified_entries);
  Serial.println(" entries.");
#endif

  return modified_entries;
}

bool NvMem::load(void)
{
  const int n = size;
  struct_is_valid = false;
  byte arr[n] = {};

  for (int i = 0; i < n; i++)
  {
#if NVMEM_INVERT
    int address = NVMEM_LENGTH - NVMEM_OFFSET - n + i;
#else
    int address = NVMEM_OFFSET + i;
#endif
    arr[i] = EEPROM.read(address);
    if (arr[i] != 0xFF)
    {
      struct_is_valid = true;
#ifdef NVMEM_DEBUG
      Serial.println("NV struct is valid!");
#endif
    }

#ifdef NVMEM_DEBUG
    Serial.print("Reading EEPROM address ");
    Serial.print(address);
    Serial.print(" with value ");
    Serial.println(arr[i]);
#endif
  }

  // set the EEPROM data to NVMEM object
  mem = deserialize(arr, n);

  // update NV if version has changed
  if (mem.version != NVMEM_VERSION)
  {
    if (mem.version == 0xFF)
    {
      struct_is_valid = false; // restore to defaults, even if other garbage is set in other values
#ifdef NVMEM_DEBUG
      Serial.println("NV struct is invalid due to 'version' == 0xFF!");
#endif
    }

    // update NVMEM struct, then save it (will fail if struct is not valid, see if above)
    else if (update())
      save();
  }

#ifdef NVMEM_DEBUG
  extern float byte_to_float(byte b); // declared elsewhere
  Serial.println("Loaded NVMEM struct values:");
  Serial.print("version: ");
  Serial.println(mem.version);
  Serial.print("pid: ");
  Serial.println(mem.pid);
  Serial.print("OffsetA: ");
  Serial.print(mem.OffsetA);
  Serial.print(" (");
  Serial.print(byte_to_float(mem.OffsetA));
  Serial.println(")");
#endif

  return struct_is_valid;
}

bool NvMem::save(void)
{
  if (!struct_is_valid)
  {
#ifdef NVMEM_DEBUG
    Serial.println("NVMEM struct is NOT VALID. Cannot save().");
#endif
    return false;
  }

  const int n = size;
  byte arr[n] = {};
  serialize(arr, mem);

#ifdef NVMEM_DEBUG
  void printArray(byte arr[], int n);
  Serial.print("Writing ");
  printArray(arr, n);
#endif

  for (int i = 0; i < n; i++)
  {
#if NVMEM_INVERT
    int address = NVMEM_LENGTH - NVMEM_OFFSET - n + i;
#else
    int address = NVMEM_OFFSET + i;
#endif
    EEPROM.update(address, arr[i]);

#ifdef NVMEM_DEBUG
    Serial.print("Writing EEPROM address ");
    Serial.print(address);
    Serial.print(" with value ");
    Serial.println(arr[i]);
#endif
  }
#if NVMEM_INVERT
  for (int i = NVMEM_LENGTH - NVMEM_OFFSET - n - 1; i >= 0; i--)
#else
  for (int i = NVMEM_OFFSET + n; i <= NVMEM_LENGTH - 1; i++)
#endif
  {
    if (EEPROM.read(i) != 0xFF)
      EEPROM.write(i, 0xFF);
    else
    {
#ifdef NVMEM_DEBUG
      Serial.print("Stopped scanning for empty EEPROM at address ");
      Serial.println(i);
#endif
      break;
    }
  }
  return true;
}

byte *NvMem::serialize(byte arr[], NvMem_RAM data)
{
  memcpy(arr, (byte *)&data, size);
#if NVMEM_INVERT
  reverse(arr, size);
#endif
  return arr;
}

NvMem_RAM NvMem::deserialize(byte *arr, size_t len)
{
#if NVMEM_INVERT
  reverse(arr, len);
#endif
  NvMem_RAM data;
  memcpy(&data, arr, len);
  return data;
}

#ifdef NVMEM_DEBUG
// Auxuliary function to print the array
void printArray(byte arr[], int n)
{
  // Serial.print("n: ");
  // Serial.println(n);
  Serial.print("Array: ");
  for (int i = 0; i < n; i++)
  {
    // Serial.print("i: ");
    // Serial.println(i);
    Serial.print(arr[i]);
    Serial.print(" ");
  }
  Serial.println();
}
#endif

// **********************************************************
// Function to do a byte swap in a byte array
void NvMem::reverse(byte *arr, size_t len)
{
#ifdef NVMEM_DEBUG
  Serial.print("Reverse() Input ");
  printArray(arr, len);
#endif
  for (uint i1 = 0; i1 < len / 2; i1++)
  {
    uint i2 = len - 1 - i1;
    if (i1 == i2)
      break; // if odd number of values, last one is the same index, skip it
    byte t = arr[i1];
    arr[i1] = arr[i2];
    arr[i2] = t;
  }
#ifdef NVMEM_DEBUG
  Serial.print("Reverse() Output ");
  printArray(arr, len);
#endif
}

bool NvMem::isValid(void)
{
  return struct_is_valid;
}