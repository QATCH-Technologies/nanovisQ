/**************************************************************************/
/*!
    @file     MCP9808.h
    @author   K. Townsend (Adafruit Industries)
	@license  BSD (see license.txt)

	This is a library for the Adafruit MCP9808 Temp Sensor breakout board
	----> http://www.adafruit.com/products/1782

	Adafruit invests time and resources providing this open source code,
	please support Adafruit and open-source hardware by purchasing
	products from Adafruit!

	@section  HISTORY

    v1.0  - First release
*/
/**************************************************************************/

#ifndef MCP9808_H
#define MCP9808_H

#if ARDUINO >= 100
#include "Arduino.h"
#else
#include "WProgram.h"
#endif
#include <Wire.h>

//#define MCP9808_DEBUG

#define MCP9808_I2CADDR_DEFAULT        0x18
#define MCP9808_REG_CONFIG             0x01

#define MCP9808_REG_CONFIG_SHUTDOWN    0x0100
#define MCP9808_REG_CONFIG_CRITLOCKED  0x0080
#define MCP9808_REG_CONFIG_WINLOCKED   0x0040
#define MCP9808_REG_CONFIG_INTCLR      0x0020
#define MCP9808_REG_CONFIG_ALERTSTAT   0x0010
#define MCP9808_REG_CONFIG_ALERTCTRL   0x0008
#define MCP9808_REG_CONFIG_ALERTSEL    0x0004
#define MCP9808_REG_CONFIG_ALERTPOL    0x0002
#define MCP9808_REG_CONFIG_ALERTMODE   0x0001

#define MCP9808_REG_UPPER_TEMP         0x02
#define MCP9808_REG_LOWER_TEMP         0x03
#define MCP9808_REG_CRIT_TEMP          0x04
#define MCP9808_REG_AMBIENT_TEMP       0x05
#define MCP9808_REG_MANUF_ID           0x06
#define MCP9808_REG_DEVICE_ID          0x07

class MCP9808 {
  public:
    MCP9808();
    boolean begin(uint8_t a = MCP9808_I2CADDR_DEFAULT);
    void shutdown_wake(uint8_t sw_ID);
    void shutdown(void);
    void wakeup(void);
    float readTempC(void);
  private:
    void write16(uint8_t reg, uint16_t val);
    uint16_t read16(uint8_t reg, uint8_t dostop = true);
    uint8_t _i2caddr;
};

#endif
