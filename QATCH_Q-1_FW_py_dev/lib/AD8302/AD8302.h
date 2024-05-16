///////////////////////////////////////////////////////////////////////////////
//
//  Analog Devices AD8302 Library for Arduino
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

#ifndef AD8302_H
#define AD8302_H

#if ARDUINO < 100
#include "WProgram.h"
#else
#include "Arduino.h"
#endif
#include <ADC.h>


class AD8302 {
  public:
    AD8302();
    void begin(int, int, int, int);
    void config(int, int, int);
    int read();
  private:
    ADC *adc;
    int _PIN;
};

#endif // #ifndef AD8302_H
