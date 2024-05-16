# Simulator.py
# A very crude simulator for PySerial assuming it
# is emulating an Arduino (or similar) device.

from QATCH.common.logger import Logger as Log
from time import time
from struct import pack
import random
import numpy as np

TAG = "[Simulator]"

# a Serial class emulator
class serial:

    STOPBITS_ONE = 1
    EIGHTBITS = 8

    ## init(): the constructor.  Many of the arguments have default values
    # and can be skipped when calling the constructor.
    def Serial( port='COM1', baudrate = 19200, timeout=1,
                bytesize = 8, parity = 'N', stopbits = 1, xonxoff=0,
                rtscts = 0, dsrdtr = 0, write_timeout = None,
                inter_byte_timeout = None ):

        return serial( port, baudrate, timeout,
                       bytesize, parity, stopbits, xonxoff,
                       rtscts, dsrdtr, write_timeout,
                       inter_byte_timeout )

    def __init__( self, port='COM1', baudrate = 19200, timeout=1,
                  bytesize = 8, parity = 'N', stopbits = 1, xonxoff=0,
                  rtscts = 0, dsrdtr = 0, write_timeout = None,
                  inter_byte_timeout = None ):
        self.name     = port
        self.port     = port
        self.timeout  = timeout
        self.parity   = parity
        self.baudrate = baudrate
        self.bytesize = bytesize
        self.stopbits = stopbits
        self.xonxoff  = xonxoff
        self.rtscts   = rtscts
        self.dsrdtr   = dsrdtr
        self.write_timeout      = write_timeout
        self.inter_byte_timeout = inter_byte_timeout
        self.is_open    = False
        self.in_waiting = 0
        self._in_data   = ""
        self._out_data  = b''

        self.__init_simulator__()

    def __init_simulator__( self ):
        self.sim_log = False
        self.start_time = time()
        Log.d (TAG, 'Simulator initialized!')

        # BUILD INFO
        self.DEVICE_BUILD = "QATCH Q-1"
        self.CODE_VERSION = "v2.2x SIMULATOR"
        self.RELEASE_DATE = "2020-07-28"

        # DEFINE I/O PINS
        '''
        // DDS Synthesizer AD9851 pin function
        #define WCLK            A8
        #define DATA            A9
        #define FQ_UD           A1
        // phase comparator AD8302 pinout
        #define AD8302_PHASE    20
        #define AD8302_MAG      37
        //#define AD83202_REF     17
        #define AD83202_REF     34
        // LED pin
        #define LED_PIN_1       24
        #define LED_PIN_2       25
        '''
        # DEFINE CONSTANTS
        '''
        // potentiometer AD5252 I2C address is 0x2C(44)
        #define ADDRESS         0x2C
        // potentiometer AD5252 default value for compatibility with openQCM Q-1 shield @5VDC
        #define POT_VALUE       240 // was 254
        // reference clock
        #define REFCLK          125000000
        // number of ADC samples to average
        #define AVERAGE_SAMPLE  1
        // in int number of averaging
        #define MAG_AVG         4
        #define PHASE_AVG       8
        // teensy ADC averaging init
        #define ADC_RESOLUTION  13
        # ADC waiting delay microseconds (only if needed)
        self.WAIT_DELAY_US = 500
        // Serial baud rate
        #define BAUD            2000000
        '''
        # Sample delay between set and read frequency when sweeping
        self.FREQ_OFFSET = 0 # sim-only

        # DEFINE OPTIONS
        '''
        // ADC averaging
        #define AVERAGING       True
        // analog to digital conversion method
        #define ARDUINO_ADC     False
        // use continuous read functions vs. one-shot (Teensy only)
        #define CONTINUOUS_ADC  True
        '''
        # use to encode serial bytes instead of sending plaintext (faster, but harder to debug)
        self.ENCODE_SERIAL = True
        # use to set default state of whether to report phase data or not
        self.REPORT_PHASE  = False
        # use to send reading deltas instead of absolute values
        self.USE_DELTAS    = False
        # use to turn on continuous streaming of data once started
        self.DO_STREAM     = True
        # use only for serial monitor debugging (will confuse application)
        self.DEBUG         = False
        '''
        // Force turn off averaging if disabled
        #if AVERAGING == False
        #undef AVERAGE_SAMPLE
        #define AVERAGE_SAMPLE  1
        #undef MAG_AVG
        #define MAG_AVG         1
        #undef PHASE_AVG
        #define PHASE_AVG       1
        #endif
        '''
        # VARIABLE DECLARATION
        '''
        // current input frequency
        long freq = 0;
        // frequency tuning word
        long FTW;
        // temporary variable
        float temp_FTW;
        // TODO
        double val = 0;

        // Create the MCP9808 temperature sensor object
        Adafruit_MCP9808 tempsensor = Adafruit_MCP9808();
        // init temperature variable
        float temperature = 0;
        '''
        # used to add ADC init delay after calling SetFreq()
        #self.waitNeeded = False # set dynamically
        self.reportTemp = True # dynamic (not in every sample)
        self.reportPhase = True # dynamic (programmable)

        # init sweep param
        self.freq_start = 0
        self.freq_stop = 0
        self.freq_step = 0
        self.freq_start_up = 0
        self.freq_stop_up = 0
        self.freq_step_up = 0
        self.freq_start_down = 0
        self.freq_stop_down = 0
        self.freq_step_down = 0
        self.base_overtones_per_cycle = 0
        self.base_overtone_counter = 0
        '''
        # init output ad8302 measurement (cast to double)
        self.measure_phase = 0
        self.measure_mag = 0

        // Object for ADC interface driver
        ADC *adc;
        '''
        # Flags for triggering and tracking serial I/O
        self.message = False
        #self.pre_time = 0
        #self.last_time = 0
        self.last_temp = 0
        #self.last_jump = 0
        '''
        int byteAtPort = 0;
        '''
        # delta enoding helpers
        self.targetDelta_mag = 0
        self.targetDelta_ph = 0
        self.lastLastVal_mag = 0
        self.lastLastVal_ph = 0
        self.lastVal_mag = 0
        self.lastVal_ph = 0
        #self.everyOther = True
        #self.firstHit = True
        self.streaming = False
        self.swSpeed_us = 0

        # Timeout timer to prevent it from staying on forever
        self.quack_counter = 0
        self.quack_interval = 100000 # 100k samples is ~10 minutes @ 6ms/s

    ## isOpen()
    # returns True if the port to the Arduino is open.  False otherwise
    def isOpen( self ):
        return self.is_open

    ## open()
    # opens the port
    def open( self ):
        self.is_open = True

    ## close()
    # closes the port
    def close( self ):
        self.is_open = False

    ## write()
    # writes a string of characters to the Arduino
    def write( self, string ):
        string = string.decode("utf-8")
        if self.sim_log: Log.d(TAG, 'RX "' + string.rstrip() + '"' )
        self._in_data += string

        self.run() # run simulator processor to handle cmd

    ## read()
    # reads n characters from the fake Arduino. Actually n characters
    # are read from the string _out_data and returned to the caller.
    def read( self, n=1 ):
        s = self._out_data[0:n]
        self._out_data = self._out_data[n:]
        self.in_waiting = len(self._out_data)

        if self.message:
            self.__message__() # append to queue if pending

        return s

    def reset_input_buffer( self ):
        self._out_data = b''

    def reset_output_buffer( self ):
        self._in_data = ""

    ## __str__()
    # returns a string representation of the serial class
    def __str__( self ):
        return  "Serial<id=0xa81c10, open=%s>( port='%s', baudrate=%d," \
               % ( str(self.isOpen), self.port, self.baudrate ) \
               + " bytesize=%d, parity='%s', stopbits=%d, xonxoff=%d, rtscts=%d,"\
               % ( self.bytesize, self.parity, self.stopbits, self.xonxoff,
                   self.rtscts ) \
               + " dsrdtr=%d, write_timeout='%s', inter_byte_timeout=%d)"\
               % ( self.dsrdtr, self.write_timeout, self.inter_byte_timeout )

    ## __readline__()
    # reads a line from the simulated serial input buffer_recv_size
    def __readline__( self ):
        returnIndex = self._in_data.find( "\n" )
        if returnIndex != -1:
            s = self._in_data[0:returnIndex]
            self._in_data = self._in_data[returnIndex+1:]
            return s.upper()
        else:
            return ""

    ## __writestr__()
    # writes a string to end of simulated serial output buffer
    def __writestr__( self, string ):
        self._out_data += string.encode()
        self.in_waiting = len(self._out_data)
        if self.sim_log: Log.d(TAG, 'TX "' + string.rstrip() + '"' )

    ## __writebyte__()
    # writes a byte to end of simulated serial output buffer
    def __writebyte__( self, byte ):
        self._out_data += pack("B", byte)
        self.in_waiting = len(self._out_data)
        if self.sim_log: Log.d(TAG, 'TX "' + "0x{0:02x}".format(byte) + '"' )

    def run( self ):
        cmd = self.__readline__()
        while not cmd == "":
            # send one byte here to give SW something to read
            # once a byte is read, additional bytes are queued by __message__
            self.quack_counter = 0 # reset stream timeout counter
            if cmd == "VERSION":
                self.__writestr__(self.DEVICE_BUILD + "\r\n")
                self.__writestr__(self.CODE_VERSION + "\r\n")
                self.__writestr__(self.RELEASE_DATE + "\r\n")
            elif cmd == "PROGRAM":
                Log.d (TAG, 'NOTICE: Cannot be programmed!')
            elif cmd.find("SPEED") == 0:
                self.swSpeed_us = int(cmd[6:])
            elif cmd == "STREAM":
                self.streaming = True
                self.message = True
            elif cmd == "STOP":
                self.streaming = False
                self.message = False
                self.last_temp = 0
                self.__writestr__("STOP\r\n")
            else:
                self.reportPhase = self.REPORT_PHASE # default (if not specified in cmd)
                # decode message
                param = cmd.split(';') # delimiter is the semicolon
                for nn in range(0, len(param)):
                    str = param[nn]
                    # frequency start
                    if (nn == 0):
                        # only WAIT if freq_start is different than last time (hopping)
                        this_freq_start = int(str)
                        self.waitNeeded = (this_freq_start != self.freq_start)
                        self.freq_start = this_freq_start
                        # Serial.print("FREQ START = ");
                        # Serial.println(freq_start);
                    # frequency stop
                    elif (nn == 1):
                        self.freq_stop = int(str)
                    # frequency step
                    elif (nn == 2):
                        self.freq_step = int(str)
                        self.message = True # mark sweep to begin!
                        if not self.streaming:
                            # reset global variables for new sweep
                            self.swSpeed_us = 0
                            self.freq_start_up = self.freq_stop_up = self.freq_step_up = 0
                            self.freq_start_down = self.freq_stop_down = self.freq_step_down = 0
                            self.base_overtones_per_cycle = self.base_overtone_counter = 0
                        self.__writestr__( 'S' ) # send Shift ACK
                    # indeterminate meaning, must check value to decide
                    elif (nn == 3):
                        val = int(str)
                        if (val == 0 or val == 1):
                            # report phase (0 no / 1 yes)
                            self.reportPhase = val
                            break # ignore any additional delimited data!
                        else:
                          # frequency start up
                          self.freq_start_up = val # cannot be 0/1
                    # frequency stop up
                    elif (nn == 4):
                        self.freq_stop_up = int(str)
                    # frequency step up
                    elif (nn == 5):
                        self.freq_step_up = int(str)
                    # frequency start down
                    elif (nn == 6):
                        self.freq_start_down = int(str)
                    # frequency stop down
                    elif (nn == 7):
                        self.freq_stop_down = int(str)
                    # frequency step down
                    elif (nn == 8):
                        self.freq_step_down = int(str)
                    # base overtones per cycle
                    elif (nn == 9):
                        self.base_overtones_per_cycle = int(str)
                        break # ignore any additional delimited data!

            cmd = self.__readline__() # read next line (if any)


    def __message__( self ):
        # buffer up to one serial sweep in out queue
        # wait until it's read before buffering more
        if self.in_waiting == 0:
            # NOTE: 'Q' already clocked out!
            # start sweep
            count = 0
            this_freq_start = self.freq_start
            this_freq_stop = self.freq_stop
            this_freq_step = self.freq_step

            # reset state variables for each sweep
            everyOther = True
            firstHit = True
            self.quack_counter += 1

            # stop stream if one-shot or "quacking" too long
            if ((self.quack_counter > self.quack_interval) or
                not(self.DO_STREAM and self.streaming)):
              if self.streaming: Log.d (TAG, 'NOTICE: Streaming turned off!')
              self.streaming = False # if asked for, but not enabled
              self.message = False # one-shot only (not repeatedly)
            '''
            if (CONTINUOUS_ADC)
            {
              // ADC start measure gain
              adc->adc1->startContinuous(AD8302_MAG);

              if (reportPhase)
              {
                // ADC start measure phase
                adc->adc0->startContinuous(AD8302_PHASE);
              }
            }

            pre_time = time()
            '''
            # do frequency hopping (if configured)
            if not (self.base_overtones_per_cycle == 0):
              self.base_overtone_counter += 1
              if (self.base_overtone_counter >= self.base_overtones_per_cycle):
                if (self.base_overtone_counter == self.base_overtones_per_cycle):
                  this_freq_start = self.freq_start_up
                  this_freq_stop = self.freq_stop_up
                  this_freq_step = self.freq_step_up
              else:
                  this_freq_start = self.freq_start_down
                  this_freq_stop = self.freq_stop_down
                  this_freq_step = self.freq_step_down

                  self.base_overtone_counter = 0 # repeat

            # simulator specific code (calculate curve)
            peak_m = 2800
            base_m = 800
            peak_f = 5055000
            bandwidth = 2500
            if (time() - self.start_time > 10):
                shift_factor = 650 # 300, 500, 650
                t = (time() - self.start_time - 15) * 10
                t = 0 if t < 0 else min(t, shift_factor)
                peak_m -= (shift_factor - t)
                base_m = 800
                peak_f -= (shift_factor - t)
                bandwidth += (shift_factor - t) * 2

            left_f = peak_f - bandwidth
            right_f = peak_f + bandwidth
            numPts = int((this_freq_stop - this_freq_start) / this_freq_step) + 1
            datPts = int((right_f - left_f) / this_freq_step)
            x_data = np.linspace(0, np.pi, datPts)
            y_data = np.sin(x_data)

            this_f = this_freq_start
            ix = 0
            self.curve = [0] * numPts

            for x in range (0, numPts):
                #self.curve[x] = x
                #continue
                if this_f < left_f:
                    self.curve[x] = base_m
                elif ix >= len(y_data):
                    self.curve[x] = base_m
                else:
                    if ix == 0:
                        ix = int((this_f - left_f) / this_freq_step)
                    self.curve[x] = base_m + (y_data[ix] * (peak_m - base_m))
                    ix += 1
                this_f += this_freq_step
            #Log.d ("curve {}".format(self.curve))

            # start sweep cycle measurement
            count = this_freq_start
            idx = 0
            while count <= this_freq_stop + (this_freq_step * self.FREQ_OFFSET):
                '''
                // set AD9851 DDS current frequency
                SetFreq(count);

                // do the magic ! waiting for the ADC measure
                // also doing serial TX here allows signal to settle
                //      some after SetFreq() and before ADC measure!
                if (count == this_freq_start)
                {
                  if (waitNeeded)
                  {
                    // We are jumping now, so wait the full time
                    //Serial.println("wait 1: 500us");
                    delayMicroseconds(WAIT_DELAY_US);
                  }
                  else // We *might* still need to wait for the last "jump" from end of last loop
                  {
                    long elapsed = (micros() - last_jump);
                    long waitFor = (WAIT_DELAY_US - elapsed);

                    if (elapsed <= 0)
                    {
                      // micros() rolled over or no time elapsed (rarely)
                      delayMicroseconds(WAIT_DELAY_US);
                    }
                    else if (waitFor > 0 && waitFor <= WAIT_DELAY_US)
                    {
                      // this is normal path, wait the remaining duration
                      delayMicroseconds(waitFor);
                    }
                  }
                '''
                if (count == this_freq_start):
                    if (self.ENCODE_SERIAL):
                        self.reportTemp = ((time() - self.last_temp) > 2)

                        # "Q" denotes system: QATCH
                        # "A" denotes format: mag, phase, temp
                        # "B" denotes format: mag, phase
                        # "C" denotes format: mag, temp
                        # "D" denotes format: mag
                        # "E" and "F" are not supported
                        # "G" denotes format: mag deltas, raw temp
                        # "H" denotes format: mag deltas only
                        self.__writestr__( 'Q' )
                        if (self.reportPhase):
                            if not (self.USE_DELTAS):
                                self.__writestr__( "A" if self.reportTemp else "B" )
                            else: # deltas
                                self.__writestr__( "E" if self.reportTemp else "F" )
                        else: # no phase
                            if not (self.USE_DELTAS):
                                self.__writestr__( "C" if self.reportTemp else "D" )
                            else: # deltas
                                self.__writestr__( "G" if self.reportTemp else "H" )

                        if (self.USE_DELTAS):
                            # report frequency hopping (if configured)
                            overtone = 0xFF
                            if not (self.base_overtones_per_cycle == 0):
                                if (this_freq_start == self.freq_start_up):
                                    overtone = 1
                                elif (this_freq_start == self.freq_start_down):
                                    overtone = 2
                                else:
                                    overtone = 0
                            self.__writebyte__(overtone)
                '''
                // measure gain phase
                app_mag = 0
                app_phase = 0

                # ADC measure and averaging
                for (int i = 0; i < AVERAGE_SAMPLE; i++)
                {
                    if (ARDUINO_ADC)
                    {
                        // ADC measure gain
                        app_mag += analogRead(AD8302_MAG);

                        if (reportPhase)
                        {
                            // ADC measure phase
                            app_phase += analogRead(AD8302_PHASE);
                        }
                    }
                    else if (CONTINUOUS_ADC)
                    {
                        while (!adc->adc1->isComplete()) yield();
                        // ADC measure gain
                        app_mag += adc->adc1->analogReadContinuous();

                        if (reportPhase)
                        {
                            while (!adc->adc0->isComplete()) yield();
                            // ADC measure phase
                            app_phase += adc->adc0->analogReadContinuous();
                        }
                    }
                    else
                    {
                        // start measurement
                        adc->adc1->startReadFast(AD8302_MAG);
                        // wait for ADC to finish
                        while (adc->adc1->isConverting()) yield();
                        // ADC measure gain
                        app_mag += adc->adc1->readSingle();

                        if (reportPhase)
                        {
                            // start measurement
                            adc->adc0->startReadFast(AD8302_PHASE);
                            // wait for ADC to finish
                            while (adc->adc0->isConverting()) yield();
                            // ADC measure phase
                            app_phase += adc->adc0->readSingle();
                        }
                    }
                }

                // averaging (cast to double)
                self.measure_mag = 1.0 * app_mag / AVERAGE_SAMPLE;

                if (reportPhase)
                {
                    measure_phase = 1.0 * app_phase / AVERAGE_SAMPLE;
                }
                '''
                if (count >= this_freq_start + (this_freq_step * self.FREQ_OFFSET)):

                    # ADC measure gain
                    measure_mag = self.curve[idx] + random.triangular(-25, 25)
                    if (self.reportPhase):
                        # ADC measure phase
                        measure_phase = self.curve[idx] * 0.7 + random.triangular(-25, 25)
                    idx += 1

                    # serial write data (all values)
                    if (self.USE_DELTAS):
                        if (firstHit):
                            mag_int = int(measure_mag)
                            mag_int0 = (mag_int & 0x00FF) >> 0
                            mag_int1 = (mag_int & 0xFF00) >> 8
                            self.__writebyte__(mag_int1)
                            self.__writebyte__(mag_int0)

                            self.targetDelta_mag = 0
                            self.lastLastVal_mag = mag_int

                            if (self.reportPhase):
                                ph_int = int(measure_phase)
                                ph_int0 = (ph_int & 0x00FF) >> 0
                                ph_int1 = (ph_int & 0xFF00) >> 8
                                self.__writebyte__(ph_int1)
                                self.__writebyte__(ph_int0)

                                self.targetDelta_ph = 0
                                self.lastLastVal_ph = ph_int

                            firstHit = False
                        elif (everyOther):
                            self.lastVal_mag = int(measure_mag)

                            if (self.reportPhase):
                                self.lastVal_ph = int(measure_phase)

                            everyOther = False
                        else:
                            delta_mag = self.__deltaMag__(self.lastVal_mag, int(measure_mag))
                            self.__writebyte__(delta_mag)

                            if (self.reportPhase):
                                delta_ph = self.__deltaPh__(self.lastVal_ph, int(measure_phase))
                                self.__writebyte__(delta_ph)

                            everyOther = True
                    elif (self.ENCODE_SERIAL):
                        mag_int = int(measure_mag)
                        mag_int0 = (mag_int & 0x00FF) >> 0
                        mag_int1 = (mag_int & 0xFF00) >> 8
                        self.__writebyte__(mag_int1)
                        self.__writebyte__(mag_int0)

                        if (self.reportPhase):
                            phase_int = int(measure_phase)
                            phase_int0 = (phase_int & 0x00FF) >> 0
                            phase_int1 = (phase_int & 0xFF00) >> 8
                            self.__writebyte__(phase_int1)
                            self.__writebyte__(phase_int0)
                    else:
                        self.__writestr__("{0:.2f}".format(measure_mag))
                        self.__writestr__( ';' )

                        if (self.reportPhase):
                            self.__writestr__("{0:.2f}".format(measure_phase))
                            self.__writestr__( '\r\n' )
                    '''
                    // if this is a cal run, and fw goes too fast for too long, bytes may get dropped by sw serial link
                    // to combat this, if this is NOT a streaming run AND it has over 500+ samples (assumed to be cal),
                    // then add a little extra delay every 500 samples to allow slower serial buses to catch up on data
                    if (!streaming)
                    {
                        if (((count - this_freq_start) / this_freq_step) % 500 == 500 - 1)
                        {
                            delay(3.3); // over a standard calibration run this should happen 200x
                        }
                    }
                }
                '''
                count += this_freq_step
            '''
            }

            // Set HW back to first freq assuming another sweep is to come
            SetFreq(this_freq_start); // TODO AJR: figure out if we're hopping and set NEXT freq, not THIS one!
            last_jump = micros();

            if (CONTINUOUS_ADC)
            {
              // ADC stop measure gain
              adc->adc1->stopContinuous();

              if (reportPhase)
              {
                // ADC stop measure phase
                adc->adc0->stopContinuous();
              }
            }
            '''
            if (self.reportTemp):
                self.last_temp = time()

                # measure temperature
                temperature = random.triangular(25.5, 26.5)

                # serial write temperature data at the end of the sweep
                if (self.ENCODE_SERIAL):
                    temp_int = int(temperature * 100)
                    temp_int0 = (temp_int & 0x00FF) >> 0
                    temp_int1 = (temp_int & 0xFF00) >> 8
                    self.__writebyte__(temp_int1)
                    self.__writebyte__(temp_int0)
                else:
                    self.__writestr__("{0:.2f}".format(temperature))
                    self.__writestr__( ';' )

            # print termination char EOM
            if not (self.ENCODE_SERIAL):
                self.__writestr__("s\r\n")
            '''
            self.last_time = time()

            if (swSpeed_us && message)
            {
              unsigned long fwSpeed_us = (last_time - pre_time);
              if (swSpeed_us > fwSpeed_us) // if micros() rolls, this will be false
              {
                unsigned long delayFor_us = (swSpeed_us - fwSpeed_us);
                delayMicroseconds(delayFor_us);
              }
              // else, serial speed is self-throttling (no action needed)
            }

            if (DEBUG)
            {
              Serial.flush();
              Serial.println();
              Serial.print("Total samples:   ");
              Serial.println(((this_freq_stop - this_freq_start) / this_freq_step) + 1);
              Serial.print("Sweep time (ms): ");
              Serial.print((last_time - pre_time) / 1000);
              Serial.print(".");
              Serial.println((last_time - pre_time) % 1000);
            }

            // end sweep
            '''
            # report stream timeout event
            if (self.quack_counter > self.quack_interval):
              self.__writestr__("QUACK!\r\n")


    # Send deltas for magnitude
    def __deltaMag__( self, val1, val2 ):
        compress = 4 # compression ratio (MUST MATCH SW)
        out = 0 # retval
        sign = 0

        #Log.d ("val0 {} val1 {} val2 {}".format(self.lastLastVal_mag, val1, val2))
        #delta1 = (val1 - self.lastLastVal_mag)
        #delta2 = (val2 - val1)
        #Log.d ("delta1 {} delta2 {}".format(delta1, delta2))
        #enc1 = enc2 = 0

        # calculate val1
        self.targetDelta_mag += (val1 - self.lastLastVal_mag)
        #Log.d ("mag1 {}".format(self.targetDelta_mag))
        if (self.targetDelta_mag > 7 * compress):
            # too big to encode now, indicate max and send rest later
            out |= 0x70 # 0111 0000 (7)
            self.targetDelta_mag -= (7 * compress)

            Log.d ("too big")

        elif (self.targetDelta_mag < -8 * compress):
            # too small to encode now, indicate min and send rest later
            out |= 0x80 # 1000 0000 (-8)
            self.targetDelta_mag -= (-8 * compress)

            Log.d ("too small")

        else:
            # entire delta can be encoded now, so do that
            out |= (int(self.targetDelta_mag / compress) << 4)
            sign = 1 if self.targetDelta_mag >= 0 else -1
            self.targetDelta_mag = (self.targetDelta_mag % (sign * compress))

            #enc1 = ((out if out >= 0 else out + 256) >> 4)
            #Log.d ("enc1 {}".format(enc1))
            #Log.d ("rem1 {}".format(self.targetDelta_mag))

        # calculate val2
        self.targetDelta_mag += (val2 - val1)
        #Log.d ("mag2 {}".format(self.targetDelta_mag))
        if (self.targetDelta_mag > 7 * compress):
            # too big to encode now, indicate max and send rest later
            out |= 0x07 # 0000 0111 (7)
            self.targetDelta_mag -= (7 * compress)

            Log.d ("too big")

        elif (self.targetDelta_mag < -8 * compress):
            # too small to encode now, indicate min and send rest later
            out |= 0x08 # 0000 1000 (-8)
            self.targetDelta_mag -= (-8 * compress)

            Log.d ("too small")

        else:
            # entire delta can be encoded now, so do that
            out |= (int(self.targetDelta_mag / compress) & 0x0F)
            sign = 1 if self.targetDelta_mag >= 0 else -1
            self.targetDelta_mag = (self.targetDelta_mag % (sign * compress))

            #enc2 = (out & 0x0F)
            #Log.d ("enc2 {}".format(enc2))
            #Log.d ("rem2 {}".format(self.targetDelta_mag))

        # store last val for next time
        self.lastLastVal_mag = val2

        # return encoded byte
        return out if out >= 0 else out + 256 # 2's complement math when out < 0


    # Send deltas for phase
    def __deltaPh__( self, val1, val2 ):
        compress = 4 # compression ratio (MUST MATCH SW)
        out = 0 # retval
        sign = 0

        #Log.d ("val0 {} val1 {} val2 {}".format(self.lastLastVal_mag, val1, val2))
        #delta1 = (val1 - self.lastLastVal_mag)
        #delta2 = (val2 - val1)
        #Log.d ("delta1 {} delta2 {}".format(delta1, delta2))
        #enc1 = enc2 = 0

        # calculate val1
        self.targetDelta_ph += (val1 - self.lastLastVal_ph)
        #Log.d ("mag1 {}".format(self.targetDelta_mag))
        if (self.targetDelta_ph > 7 * compress):
            # too big to encode now, indicate max and send rest later
            out |= 0x70 # 0111 0000 (7)
            self.targetDelta_ph -= (7 * compress)

            Log.d ("too big")

        elif (self.targetDelta_ph < -8 * compress):
            # too small to encode now, indicate min and send rest later
            out |= 0x80 # 1000 0000 (-8)
            self.targetDelta_ph -= (-8 * compress)

            Log.d ("too small")

        else:
            # entire delta can be encoded now, so do that
            out |= (int(self.targetDelta_ph / compress) << 4)
            sign = 1 if self.targetDelta_ph >= 0 else -1
            self.targetDelta_ph = (self.targetDelta_ph % (sign * compress))

            #enc1 = ((out if out >= 0 else out + 256) >> 4)
            #Log.d ("enc1 {}".format(enc1))
            #Log.d ("rem1 {}".format(self.targetDelta_mag))

        # calculate val2
        self.targetDelta_ph += (val2 - val1)
        #Log.d ("mag2 {}".format(self.targetDelta_mag))
        if (self.targetDelta_ph > 7 * compress):
            # too big to encode now, indicate max and send rest later
            out |= 0x07 # 0000 0111 (7)
            self.targetDelta_ph -= (7 * compress)

            Log.d ("too big")

        elif (self.targetDelta_ph < -8 * compress):
            # too small to encode now, indicate min and send rest later
            out |= 0x08 # 0000 1000 (-8)
            self.targetDelta_ph -= (-8 * compress)

            Log.d ("too small")

        else:
            # entire delta can be encoded now, so do that
            out |= (int(self.targetDelta_ph / compress) & 0x0F)
            sign = 1 if self.targetDelta_ph >= 0 else -1
            self.targetDelta_ph = (self.targetDelta_ph % (sign * compress))

            #enc2 = (out & 0x0F)
            #Log.d ("enc2 {}".format(enc2))
            #Log.d ("rem2 {}".format(self.targetDelta_mag))

        # store last val for next time
        self.lastLastVal_ph = val2

        # return encoded byte
        return out if out >= 0 else out + 256 # 2's complement math when out < 0
