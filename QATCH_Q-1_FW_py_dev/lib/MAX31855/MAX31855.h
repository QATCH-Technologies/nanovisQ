// this library is public domain. enjoy!
// https://learn.adafruit.com/thermocouple/

#ifndef ADAFRUIT_MAX31855_H
#define ADAFRUIT_MAX31855_H

#include "Arduino.h"
// #include <EEPROM.h>

#define CAL_FACTOR 20.0
#define FAIL_THRESHOLD 10
#define CONVERSION_RATE 100  // (ms) from MAX31855 datasheet
#define SPURIOUS_DELTA 5.0   // greater deltas are spurious
#define SPURIOUS_STALE 3000  // 3 sec hysteresis period
#define TEMP_RESOLUTION 0.25 // for showing on GUI and LCD

// MAX driver status codes:
#define MAX_STATUS_OK 0
#define MAX_FAULT_OPEN 1
#define MAX_FAULT_GND 2
#define MAX_FAULT_VCC 3
// RESERVED: UNKNOWN    4 - 7
#define MAX_FAULT_LOW 254
#define MAX_FAULT_HIGH 255

/**************************************************************************/
/*!
    @brief  Class for communicating with thermocouple sensor
*/
/**************************************************************************/
class MAX31855
{

public:
  MAX31855(int8_t SCLK, int8_t CS, int8_t MISO, int8_t WAIT);
  bool begin(void);
  bool getType(void);
  String getType(bool parsed);
  uint8_t status(void);
  String status(bool parsed);
  String error(void);
  // void EEPROM_read(void);
  float readCelsius(void);
  float readFahrenheit(void);
  float readInternal(void);
  float readCelsius(bool refresh);
  float readFahrenheit(bool refresh);
  float readInternal(bool refresh);
  void setOffsetA(float t); // A = always (FW fixed offset, in EEPROM)
  void setOffsetB(float t); // B = both (SW defined offset, heat/cool)
  void setOffsetC(float t); // C = cool (SW defined offset, cool only)
  void setOffsetH(float t); // H = heat (SW defined offset, heat only)
  void setOffsetM(float t); // M = measure (FW fixed offset, in EEPROM)
  float getOffsetA(void);
  float getOffsetB(void);
  float getOffsetC(void);
  float getOffsetH(void);
  float getOffsetM(void);
  void useOffsetM(bool enable);
  void setMode(int8_t m);
  int8_t getMode(void);
  //    bool simulate_err = false;

private:
  bool hw_is_MAX6675;
  int8_t sclk, miso, cs, wait;
  uint8_t update(void);
  uint8_t update(bool detect_hw);
  uint8_t spiread(void);
  uint8_t parseread(void);
  uint8_t parseread(bool detect_hw);
  String statushelper(uint8_t status_code);
  bool is_spurious_read(void);
  uint8_t _status = MAX_STATUS_OK;
  float _internal;
  float _temperature;
  float offsetA = 0, offsetB = 0, offsetC = 0, offsetH = 0, offsetM = 0;
  bool enable_OffsetM = false;
  int8_t mode = 0;
  uint8_t _cached_err = MAX_STATUS_OK;
  uint8_t failure_count = 0, failure_threshold = FAIL_THRESHOLD;
  unsigned long timeOfLastRead = 0;
  float _last_temperature = 0;
};

#endif
