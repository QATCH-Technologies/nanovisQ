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

#include "AD8302.h"


AD8302::AD8302 () {
}

void AD8302::begin(int phase, int mag36, int mag41, int v_ref)
{
  pinMode(phase, INPUT);
  pinMode(mag36, INPUT);
  pinMode(mag41, INPUT);
  pinMode(v_ref, INPUT);
}

void AD8302::config(int resolution, int averaging, int input)
{
  // Initialize ADC library
  adc = new ADC();

  // Teensy 3.6 set adc for mag
  adc->adc1->setResolution(resolution);
  //adc->adc1->setReference(ADC_REFERENCE::REF_3V3);
  //adc->adc1->enableCompare(1.0/3.3*adc->adc1->getMaxValue(), 0); // measurement will be ready if value < 1.0V
  //adc->adc1->enableCompareRange(1.0*adc->adc1->getMaxValue()/3.3, 2.0*adc->adc1->getMaxValue()/3.3, 0, 1); // ready if value lies out of [1.0,2.0] V
  adc->adc1->setConversionSpeed(ADC_CONVERSION_SPEED::HIGH_SPEED); // change the conversion speed to enforce 20MHz (ADLPC=1, ADHSC=0)
  adc->adc1->setSamplingSpeed(ADC_SAMPLING_SPEED::HIGH_SPEED); //
  adc->adc1->setAveraging(averaging);

  _PIN = input;
}

int AD8302::read()
{
  // start measurement
  adc->adc1->startReadFast(_PIN);
  // wait for ADC to finish
  while (adc->adc1->isConverting()) yield();
  // ADC measure gain
  return adc->adc1->readSingle();
}

// end of ad8302.cpp
