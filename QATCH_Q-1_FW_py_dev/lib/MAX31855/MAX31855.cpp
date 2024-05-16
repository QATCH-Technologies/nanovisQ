// this library is public domain. enjoy!
// https://learn.adafruit.com/thermocouple/

// modified for use with QATCH devices by Alex Ross on 8/23/2021

#include "MAX31855.h"

/**************************************************************************/
/*!
    @brief  Initialize a MAX31855 sensor
    @param   SCLK The Arduino pin connected to Clock
    @param   CS The Arduino pin connected to Chip Select
    @param   MISO The Arduino pin connected to Data Out
*/
/**************************************************************************/
MAX31855::MAX31855(int8_t SCLK, int8_t CS, int8_t MISO, int8_t WAIT)
{
  sclk = SCLK;
  cs = CS;
  miso = MISO;
  wait = WAIT; // original library used '10' us, but zero works too!

  // Serial.printf("SCLK = %u\n", SCLK);
  // Serial.printf("MISO = %u\n", MISO);

  // EEPROM_read();

  // define pin modes
  pinMode(cs, OUTPUT);
  pinMode(sclk, OUTPUT);
  pinMode(miso, INPUT_PULLUP);

  digitalWrite(cs, HIGH);
}

bool MAX31855::begin(void)
{
  // Serial.println("BEGIN!");
  update(true);
  // Serial.printf("ret = %u\n", status());
  return hw_is_MAX6675;
}

bool MAX31855::getType(void)
{
  return hw_is_MAX6675;
}

String MAX31855::getType(bool parsed)
{
  if (!parsed)
    return String(hw_is_MAX6675);

  return hw_is_MAX6675 ? "MAX6675" : "MAX31855";
}

uint8_t MAX31855::status(void)
{
  return status(false).toInt();
}
String MAX31855::status(bool parsed)
{
  if (!parsed)
    return String(_status);
  return statushelper(_status);
}

String MAX31855::error(void)
{
  String str = statushelper(_cached_err);
  if (_cached_err != MAX_STATUS_OK)
    str += " (" + String(_cached_err) + ")";
  return str;
}

String MAX31855::statushelper(uint8_t status_code)
{
  String str;
  switch (status_code)
  {
  case MAX_STATUS_OK: // 0
    str = "OK[NONE]";
    break;
  case MAX_FAULT_OPEN: // 1
    str = "FAULT[PROBE_OPEN]";
    break;
  case MAX_FAULT_GND: // 2
    str = "FAULT[PROBE_GND]";
    break;
  case MAX_FAULT_VCC: // 3
    str = "FAULT[PROBE_VCC]";
    break;
  case MAX_FAULT_LOW: // 254
    str = "FAULT[BUS_LOW]";
    break;
  case MAX_FAULT_HIGH: // 255
    str = "FAULT[BUS_HIGH]";
    break;
  default:
    str = "UNKNOWN";
    break;
  }
  return str;
}

bool MAX31855::is_spurious_read(void)
{
  bool bad_read = true;
  bad_read &= abs(_temperature - _last_temperature) > SPURIOUS_DELTA;
  bad_read &= _last_temperature != 0;
  bad_read &= millis() - timeOfLastRead < SPURIOUS_STALE;
  return bad_read;
}

/// @note main setup() must init setOffsetA() from NVMEM at boot time
// void MAX31855::EEPROM_read(void)
// {
//   // calibration offset
//   int old_address = 1; // OffsetA
//   int new_address = (EEPROM.length() - 1) - old_address;
//   byte oB = EEPROM.read(new_address);
//   if (oB == 0xFF) oB = 0; // handle special value (unset)
//   signed char sB = oB;
//   if (sB & 0x80) sB = -(~sB); // convert unsigned to signed (don't add 1)
//   // convert raw offset byte to degrees C
//   offsetA = (sB / CAL_FACTOR); // 0.05x resolution

//   if (millis() > 1000)
//   {
//     Serial.printf("Set offsetA = %2.2f\n", getOffsetA());
//   }
// }

/**************************************************************************/
/*!
    @brief  Read the Celsius temperature
    @returns Temperature in C or NAN on failure!
*/
/**************************************************************************/
float MAX31855::readCelsius(void)
{
  return readCelsius(true);
}
float MAX31855::readCelsius(bool refresh)
{
  float o = offsetA + offsetB; // EEPROM (always) and SW offset (both)

  if (refresh)
    update();

  if (_status != MAX_STATUS_OK && failure_count >= failure_threshold)
  {
    // thermocouple attachment issues!
    return NAN;
  }
  if (is_spurious_read())
  {
    failure_count++;
    // Serial.print("Read is bad. Flag incremented to: ");
    // Serial.println(failure_count);
    return _last_temperature;
  }
  _last_temperature = _temperature;
  // Serial.print("Last temperature is now: ");
  // Serial.println(_last_temperature);

  if (mode < 0) // cool
  {
    o += offsetC;
  }
  if (mode > 0) // heat
  {
    o += offsetH;
  }
  if (enable_OffsetM) // measure
  {
    o += offsetM;
  }

  timeOfLastRead = millis();

  return _temperature + o;
}

/**************************************************************************/
/*!
    @brief  Read the Fahenheit temperature
    @returns Temperature in F or NAN on failure!
*/
/**************************************************************************/
float MAX31855::readFahrenheit(void)
{
  return readFahrenheit(true);
}
float MAX31855::readFahrenheit(bool refresh)
{
  return readCelsius(refresh) * 9.0 / 5.0 + 32;
}

float MAX31855::readInternal(void)
{
  return readInternal(true);
}
float MAX31855::readInternal(bool refresh)
{
  if (refresh)
    update();
  return _internal;
}

uint8_t MAX31855::update(void)
{
  return update(false);
}
uint8_t MAX31855::update(bool detect_hw)
{
  uint8_t ret;
  bool s = digitalRead(sclk);

  if (millis() - timeOfLastRead < CONVERSION_RATE && timeOfLastRead != 0)
  {
    delay(CONVERSION_RATE); // wait for next sample
  }

  if (s)
  {
    digitalWrite(sclk, LOW);
    delay(wait); // bus reset, otherwise CS might be missed
  }

  digitalWrite(sclk, HIGH);
  digitalWrite(cs, LOW);
  delayMicroseconds(wait);

  ret = parseread(detect_hw);

  digitalWrite(cs, HIGH);
  digitalWrite(sclk, s);

  return ret;
}

uint8_t MAX31855::parseread(void)
{
  return parseread(false);
}
uint8_t MAX31855::parseread(bool detect_hw)
{
  // return value of _spiread()
  // BITS     DESCRIPTION
  // ----------------------
  // 00 - 02  STATUS
  //      03  RESERVED
  // 04 - 15  INTERNAL
  //      16  FAULT-BIT
  //      17  RESERVED
  // 18 - 30  TEMPERATURE (RAW)
  //      31  SIGN

  // set _cached_err if last read _status failed
  if (_status != MAX_STATUS_OK)
  {
    _cached_err = _status;
    failure_count++;
  }

  uint32_t value, temp1, temp2;

  value = spiread();
  value <<= 8;
  value |= spiread();

  if (hw_is_MAX6675 && !detect_hw)
  {
    _status = MAX_STATUS_OK;
    if (value & 0x4)
    {
      // no thermocouple attached!
      _status = (value & 0x4);
      return _status;
    }
    _temperature = (value >> 5) + (((value >> 3) & 0x3) * 0.25);
    return _status;
  }

  value <<= 8;
  value |= spiread();
  value <<= 8;
  value |= spiread();

  if (value == 0x00000000)
  {
    // bus is being held low, even with internal pull on on MISO pin
    // Most likely cause: VCC must be missing on MAX chip, with GND pulling the bus down
    _status = MAX_FAULT_LOW;
    return _status;
  }
  if (value == 0xFFFFFFFF) // needs a pull up on MISO pin to work properly!
  {
    // bit 3 and bit 17 should always be 0 - P10 datasheet
    // Most likely cause: GND to MAX chip is missing or data out (DO) is disconnected (OMG, no one is driving the bus!)
    _status = MAX_FAULT_HIGH; // STATUS_NO_COMMUNICATION;
    return _status;
  }
  //  if (simulate_err)
  //  {
  //    _status = MAX_FAULT_OPEN;
  //    return _status;
  //  }

  //_lastTimeRead = millis();

  //  process status bit 0-2
  _status = value & 0x0007;
  // if (_status != STATUS_OK)  // removed in 0.4.0 as internal can be valid.
  // {
  //   return _status;
  // }

  value >>= 3;

  // reserved bit 3, always 0
  value >>= 1;

  // process internal bit 4-15
  temp1 = (value & 0x07FF);
  _internal = temp1 * 0.0625;
  // negative flag set ?
  if (value & 0x0800)
  {
    _internal = -128 + _internal;
  }
  value >>= 12;

  // Fault bit ignored as we have the 3 status bits
  // _fault = value & 0x01;
  value >>= 1;

  // reserved bit 17, always 0
  value >>= 1;

  if (_status == MAX_STATUS_OK) // only if value is valid
  {
    // process temperature bit 18-30 + sign bit = 31
    temp2 = (value & 0x1FFF);
    _temperature = temp2 * 0.25;
    // negative flag set ?
    if (value & 0x2000)
    {
      _temperature = -2048 + _temperature;
    }
    if (is_spurious_read())
    {
      // do nothing here: failure_count increased in readCelsius()
      // Serial.println("Read is bad. Flag persists.");
    }
    else
    {
      failure_count = 0; // clear on good read
      // Serial.println("Read is good. Flag cleared.");
    }
    // Serial.print("RAW temp read is: ");
    // Serial.println(_temperature);
  }

  if (detect_hw)
  {
    // check if this is a MAX31855
    if (((temp1 & 0xFE00) == 0) &&
        ((temp1 & 0x003F) == (temp2 >> 2)))
    {
      hw_is_MAX6675 = true;
    }
    else
    {
      hw_is_MAX6675 = false;
    }
    // the above detection logic fails when internal/external temps are the same
    hw_is_MAX6675 = false; // force to MAX31855, making MAX6675 obsolete
  }

  //  if (hw_is_MAX6675)
  //  {
  //    // check if no thermocouple attached!
  //    _status = (temp2 & 0x1);
  //    // hack for MAX31855 reverse compat.!
  //    _temperature = (temp2 >> 1) * 0.25;
  //    _internal = NAN;
  //  }

  return _status;
}

byte MAX31855::spiread(void)
{
  int i;
  byte d = 0;

  for (i = 7; i >= 0; i--)
  {
    digitalWrite(sclk, LOW);
    delayMicroseconds(wait);

    if (digitalRead(miso))
    {
      d |= (1 << i);
    }

    digitalWrite(sclk, HIGH);
    delayMicroseconds(wait);
  }

  return d;
}

void MAX31855::setOffsetA(float t)
{
  // byte sB;
  // if (t < 0)
  // {
  if (t < -6.35)
    t = -6.35;
  // sB = ~(byte)(-t * CAL_FACTOR); // don't add 1
  // }
  // else
  // {
  if (t > 6.35)
    t = 6.35;
  // sB = (t * CAL_FACTOR);
  // }
  offsetA = t; // store to RAM, clipped if out-of-range
  /// @note parent caller must now set NVMEM and save it
  // int old_address = 1; // OffsetA
  // int new_address = (EEPROM.length() - 1) - old_address;
  // EEPROM.update(new_address, sB); // only write if different
}

void MAX31855::setOffsetB(float t)
{
  offsetB = t;
}

void MAX31855::setOffsetC(float t)
{
  offsetC = t;
}

void MAX31855::setOffsetH(float t)
{
  offsetH = t;
}

void MAX31855::setOffsetM(float t)
{
  if (t < -6.35)
    t = -6.35;
  if (t > 6.35)
    t = 6.35;
  offsetM = t; // store to RAM, clipped if out-of-range
  /// @note parent caller must now set NVMEM and save it
}

float MAX31855::getOffsetA(void)
{
  return offsetA;
}

float MAX31855::getOffsetB(void)
{
  return offsetB;
}

float MAX31855::getOffsetC(void)
{
  return offsetC;
}

float MAX31855::getOffsetH(void)
{
  return offsetH;
}

float MAX31855::getOffsetM(void)
{
  return offsetM;
}

void MAX31855::useOffsetM(bool enable)
{
  enable_OffsetM = enable;
}

void MAX31855::setMode(int8_t m)
{
  mode = m;
}

int8_t MAX31855::getMode(void)
{
  return mode;
}
