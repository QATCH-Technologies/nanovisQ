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

#include "AD9851.h"

AD9851::AD9851()
{
}

void AD9851::begin(int data, int clock, int load, unsigned long refclk, bool _6x_refclk_enable)
{
#if defined AD9851_DEBUG
  Serial.println("[AD9851] Begin");
#endif

  _DATA = data;
  _WCLK = clock;
  _FQUD = load;

  pinMode(_DATA, OUTPUT);
  pinMode(_WCLK, OUTPUT);
  pinMode(_FQUD, OUTPUT);

  _REFCLK = refclk;
  _6X_REFCLK_EN = _6x_refclk_enable;

  // Adjust REFCLK frequency when multiplier enabled
  if (_6X_REFCLK_EN)
    _REFCLK *= 6;

  // enable serial mode
  wakeup();
}

long AD9851::setFreq(long freq)
{
  _update(freq); // send it
  return freq;
}

void AD9851::shutdown()
{
  // select "Power down" control bit
  _update(1, true); // send it
}

void AD9851::wakeup()
{
#if defined AD9851_DEBUG
  Serial.println("[AD9851] Wakeup");
#endif

  // AD9851 enter serial mode
  digitalWrite(_WCLK, HIGH);
  digitalWrite(_WCLK, LOW);
  digitalWrite(_FQUD, HIGH);
  digitalWrite(_FQUD, LOW);
  delayMicroseconds(5);
}

void AD9851::_update(long freq, bool power_down) // write 32+8=40 bits to DDS
{
#if defined AD9851_DEBUG
  if (power_down)
    Serial.println("[AD9851] Shutdown");
  else
  {
    Serial.print("[AD9851] Update freq = ");
    Serial.println(freq);
  }
#endif

  // set to 125 MHz internal clock
  long FTW = long((freq * pow(2, 32)) / _REFCLK);
  long pointer = 1;
  int pointer2 = 0b10000000;
  int sleepcmd = 0b00100000;
  int refclk6x = 0b10000000;

  // 32 bit dds tuning word frequency instructions
  for (int i = 0; i < 32; i++)
  {
    if ((FTW & pointer) > 0)
      digitalWrite(_DATA, HIGH);
    else
      digitalWrite(_DATA, LOW);
    digitalWrite(_WCLK, HIGH);
    digitalWrite(_WCLK, LOW);
    pointer = pointer << 1;
  }

  // 8 bit dds phase and x6 multiplier refclock
  for (int i = 0; i < 8; i++)
  {
    if (power_down && (sleepcmd & pointer2) > 0)
      digitalWrite(_DATA, HIGH); // assert Power-Down bit
    else if (_6X_REFCLK_EN && (refclk6x & pointer2) > 0)
      digitalWrite(_DATA, HIGH); // assert 6x REFCLK Multiplier bit
    else
      digitalWrite(_DATA, LOW);
    digitalWrite(_WCLK, HIGH);
    digitalWrite(_WCLK, LOW);
    pointer2 = pointer2 >> 1;
  }

  digitalWrite(_FQUD, HIGH);
  digitalWrite(_FQUD, LOW);
}

// end of ad9851.cpp
