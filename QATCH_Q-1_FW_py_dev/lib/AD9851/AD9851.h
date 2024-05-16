///////////////////////////////////////////////////////////////////////////////
//
//  Analog Devices AD9851 Library for Arduino
//  Copyright (c) 201t Roger A. Krupski <rakrupski@verizon.net>
//
//  Last update: 04 May 2017
//
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program. If not, see <http://www.gnu.org/licenses/>.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef AD9851_H
#define AD9851_H

#if ARDUINO < 100
#include "WProgram.h"
#else
#include "Arduino.h"
#endif

// #define AD9851_DEBUG

// defines
#define CAL_VAL (0UL)                      // AJR TODO: Calculate crystal calibration offset from NTP timestamps
#define POWERDOWN_EN (1 << 2)              // enable powerdown bit
#define REFCLK_30 (30000000UL + CAL_VAL)   // reference clock 30 MHz plus calibration
#define REFCLK_125 (125000000UL + CAL_VAL) // reference clock 125 MHz plus calibration
#define REFCLK_180 (180000000UL + CAL_VAL) // reference clock 180 MHz plus calibration
// #define F_FACTOR (4294967296.0 / REF_CLK)  // frequency to binary factor
// #define P_FACTOR (360.0 / 32.0)            // phase to binary factor

class AD9851
{
public:
  AD9851();
  void begin(int, int, int, unsigned long, bool);
  int setPhase(int);
  long setFreq(long);
  void wakeup(void);
  void shutdown(void);
  void calibrate(void); // AJR TODO: Not implemented

  // REFCLK frequencies:
  const static unsigned long REFCLK_30MHz = REFCLK_30;
  const static unsigned long REFCLK_125MHz = REFCLK_125;
  const static unsigned long REFCLK_180MHz = REFCLK_180;

private:
  void _update(long, bool power_down = false);
  int _WCLK;
  int _DATA;
  int _FQUD;
  unsigned long _REFCLK;
  bool _6X_REFCLK_EN;
};

#endif // #ifndef AD9851_H
