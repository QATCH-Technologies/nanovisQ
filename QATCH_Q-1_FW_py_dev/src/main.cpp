/***********************************************************************************************
   LICENSE
   Copyright (C) 2020 QATCH Technologies LLC
   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   You should have received a copy of the GNU General Public License
   along with this program.  If not, see http://www.gnu.org/licenses/gpl-3.0.txt
  ----------------------------------------------------------------------------------------------
   OPENQCM Q-1 - Quartz Crystal Microbalance with dissipation monitoring
   openQCM is the unique opensource quartz crystal microbalance http://openqcm.com/
   ELECTRONICS
     - board and firmware designed for teensy 3.6 development board
     - DDS/DAC synthesizer AD9851
     - phase comparator AD8302
     - I2C digital potentiometer AD5251+
     - MCP9808 on-board temperature sensor
     - MAX31855 or MAX6675 temperature sensor
     - L298NHB TEC temperature setpoint
   See "Build Info" below for build device/version/date...
   author  Alexander J. Ross
   owner   QATCH Technologies, LLC
   ----------------------------------------------------------------------------------------------
   CHANGES:
   - See changelog PDF file located in ZIP folder of release
   ----------------------------------------------------------------------------------------------
   LIBRARIES:
   - ADC v9.0
   https://github.com/pedvide/ADC
   - TeensyID v1.3.1
   https://github.com/sstaub/TeensyID
   - SevenSegmentDisplay v1.0.0
   https://github.com/alikabeel/Letters-and-Numbers-Seven-Segment-Display-Library
   - FlasherX v2.1
   https://github.com/joepasquariello/FlasherX
   NOTE: These libraries are REQUIRED to build the project locally!
   ----------------------------------------------------------------------------------------------
   CREDIT:
   - based on the work made by Brett Killion on Hackaday
   https://hackaday.io/project/10021-arduino-network-analyzer
   - Teensy boards are developed by Paul Stoffregen at PJRC.com
   https://www.pjrc.com/store/teensy36.html
   - MCP9808 I2c temperature sensor driver is developed by Adafruit
   - MAX31855 I2c temperature sensor driver adapted from Adafruit example code
   Written by Kevin Townsend/Limor Fried for Adafruit Industries.
 ***********************************************************************************************/

/************************* BUILD INFO **************************/

// Build Info can be queried serially using command: "VERSION"
#define DEVICE_BUILD "QATCH Q-1"
#define CODE_VERSION "v2.6r63"
#define RELEASE_DATE "2025-10-09"

/************************** LIBRARIES **************************/

#include <Arduino.h>
// #include <EEPROM.h>
#include <TeensyID.h>
// #include <Wire.h>

#include "main.h"
#include "FlasherX.h"

#include "AD8302.h"
#include "AD9851.h"
#include "L298NHB.h"
#include "MAX31855.h"
#include "MCP9808.h"
#include "NvMem.h"

#include "ILI9341_t3.h" // includes <SPI.h>
#include "font_Poppins-Bold.h"
#include "icons.h"

/*********************** DEFINE I/O PINS ***********************/

// #define NET_TCP_DEBUG

// DDS synthesizer AD9851 pin function
#define WCLK 22
#define DATA 23
#define FQ_UD 15
// phase comparator AD8302 pinout
#define AD8302_PHASE A6
#define AD8302_MAG36 37  // AJR different on T4.1 (use A14, from pin 37 to 38)
#define AD8302_MAG41 A14 // AJR different on T4.1 (use A14, from pin 37 to 38)
// #define AD8302_REF      A3
#define AD8302_REF A15 // AJR different on T4.1 but this is not used in this sketch
// LED pin
#define LED_RED_PIN 24             // Red LED on openQCM PCB (see note below)
#define LED_BLUE_PIN 25            // Blue LED on openQCM PCB (see note below)
#define LED_ORANGE_PIN LED_BUILTIN // Orange LED on Teensy
#define LED_WHITE_PIN_0 11
#define LED_WHITE_PIN_1 255 // micro_led moved to pin 5, but this might conflict, so disable
#define LED_SEGMENT_DP 33   // currently only used by HW Rev0, but could be used on Rev1 too
// NOTE: Both LED_RED_PIN and LED_BLUE_PIN must be high for Blue LED to illuminate on openQCM boards
// L298NHB pins
#define L298NHB_M1 1                     // motor signal pin 1 (0=forward,1=back)
#define L298NHB_E1 2                     // motor enable pin 1 (PWM)
#define L298NHB_M2 3                     // motor signal pin 2 (0=forward,1=back)
#define L298NHB_E2 4                     // motor enable pin 2 (PWM)
#define L298NHB_HEAT 0                   // heat when signal is forward
#define L298NHB_COOL 1                   // cool when signal is reverse
#define L298NHB_INIT 1                   // initial PWM enable power
#define L298NHB_AUTOOFF (1000 * 60 * 10) // time to auto off (in millis): 1 min buffer between end of cooldown and start of screensaver
#define L298NHB_COOLDOWN (1000 * 60 * 4) // time to cooldown (in millis)
// MAX31855 pins
#define MAX31855_SO_0 12  // serial out
#define MAX31855_SO_1 16  // serial out
#define MAX31855_CS 10    // chip select
#define MAX31855_CLK_0 13 // serial clock
#define MAX31855_CLK_1 17 // serial clock
#define MAX31855_WAIT 0   // signal settle delay (in us)
// TEMP_CIRCUIT pin
#define TEMP_CIRCUIT 5 // relay pin for reject fan (on/off)
#define TEC_SENSE 41      // now: follow pin 5; future use: TEC_SENSE
// #define FAN_HIGH_LOW 6 // relay pin for reject fan (speed) (unused)
// Pins for ILI9341 TFT Touchscreen connections:
#define TFT_DC 21
#define TFT_CS 9
#define TFT_RST 255 // 255 = unused, connect to 3.3V
#define TFT_MOSI 11
#define TFT_SCLK 13
#define TFT_MISO 12
// Pin for reading external 5V voltage ADC
#define PIN_EXT_5V_VOLTAGE A16

/*********************** DEFINE CONSTANTS **********************/

// potentiometer AD5252 I2C address is 0x2C(44)
#define POT_ADDRESS 0x2C
// potentiometer AD5252 I2C channel
#define POT_CHANNEL 0x01
// potentiometer AD5252 default value for compatibility with openQCM Q-1 shield @5VDC
#define POT_VALUE 240 // was 254
// number of ADC samples to average
#define AVERAGE_SAMPLE 1
// in int number of averaging
#define MAG_AVG 4
#define PHASE_AVG 8
#define TEMP_AVG 4
// teensy ADC averaging init
#define ADC_RESOLUTION 13
// Serial baud rate
#define BAUD 2000000

// TFT screen resolution
#define TFT_ROTATION 1
#define TFT_WIDTH tft.width()
#define TFT_HEIGHT tft.height()
#define ICON_X 110
#define ICON_Y 32
#define TEXT_X 32
#define TEXT_Y 140

// TFT branding colors
#define QATCH_GREY_BG CL(0xF6, 0xF6, 0xF6)
#define QATCH_BLUE_FG CL(0x00, 0xA3, 0xDA)

// Define access to NVMEM options
#define NVMEM nv.mem
#define HW_REV_MATCH(t) (NVMEM.HW_Revision == t)
#define PID_IN_RANGE(pid, lo, hi) (pid >= lo && pid <= hi)
#define PID_IS_SECONDARY(pid) (PID_IN_RANGE(pid, 0x2, 0x4) || PID_IN_RANGE(pid, 0xB, 0xD))
// NOTE: Primary devices are PID = 0x00, 0x01, 0x0A, and/or 0xFF (default, when other)
// NOTE: Secondary devices are PID = 0x02, 0x03, 0x04, 0x0B, 0x0C and/or 0x0D

// Determine external voltage condition
#define L298NHB_VOLTAGE_EXPECTED 5.0                    // volts
#define L298NHB_VOLTAGE_DEVIATION 1.0                   // volts
#define L298NHB_VOLTAGE_CONVERT(adc) (9.9 * adc) / 1024 // adc -> volts
#define L298NHB_VOLTAGE_VALID(v) (abs(L298NHB_VOLTAGE_EXPECTED - v) < L298NHB_VOLTAGE_DEVIATION)

double freq_factor = 1.0;

/************************ DEFINE OPTIONS ***********************/

// ADC averaging
#define AVERAGING true
// analog to digital conversion method
#define ARDUINO_ADC false
// use continuous read functions vs. one-shot (Teensy only)
#define CONTINUOUS_ADC false
// use to encode serial bytes instead of sending plaintext (faster, but harder to debug)
#define ENCODE_SERIAL true
// use to set default state of whether to report phase data or not
#define REPORT_PHASE false
// use to send reading deltas instead of absolute values
#define USE_DELTAS true
// use to turn on continuous streaming of data once started
#define DO_STREAM true
// use only for serial monitor debugging (will confuse application)
#define DEBUG false
// use L298N H-Bridge for controlling TEC
#define USE_L298NHB true
// use MCP9808 or MAX31855 sensor for temperature readings
#define USE_MCP9808 false // do not include in build! debug only!
#define USE_MAX31855 true
// use ILI9341 TFT touchscreen driver
#define USE_ILI9341 true

// // Modify MAX31855 pins if ILI9341 missing
// #if !USE_ILI9341
// #undef MAX31855_SO
// // #undef MAX31855_CS
// #undef MAX31855_CLK
// #define MAX31855_SO 12 // serial out (old HW)
// // #define MAX31855_CS   10  // chip select (no change)
// #define MAX31855_CLK 13 // serial clock (old HW)
// #undef LED_WHITE_PIN
// #define LED_WHITE_PIN 11 // micro led (old HW)
// #endif

// Validate temperature hardware selection
#if !(USE_MCP9808 || USE_MAX31855)
#error No temperature sensor selected.
#endif
#if (USE_MCP9808 && USE_MAX31855)
#warning Both temperature sensors selected.
#endif

// Force turn off averaging if disabled
#if AVERAGING == false
#undef AVERAGE_SAMPLE
#define AVERAGE_SAMPLE 1
#undef MAG_AVG
#define MAG_AVG 1
#undef PHASE_AVG
#define PHASE_AVG 1
#endif

// Auto-Detect Teensy HW Board Type
#define TEENSY36 36
#define TEENSY41 41
#define HW_OTHER '?'
#define HW_MATCH(t) (BOARD_TYPE == t)

#if defined ARDUINO_TEENSY36
#define BOARD_TYPE TEENSY36
#define AD8302_MAG AD8302_MAG36
#elif defined ARDUINO_TEENSY41
#define BOARD_TYPE TEENSY41
#define AD8302_MAG AD8302_MAG41
#else // all other HW is unknown
#define BOARD_TYPE HW_OTHER
#define AD8302_MAG AD8302_MAG36 // ?
#endif

/************************ TEENSY41 INIT ************************/

#if HW_MATCH(TEENSY41)

#include <NativeEthernet.h>
#include <NativeEthernetUdp.h>
#include <fnet.h> // low-level servicing

// Time Sync
#define TS_SYNC_PERIOD 1000 * 60 * 60 * 1 // How often to resync time (ms)
#define TS_RETRY_COUNT 10                 // If sync fails, try this many times
#define TS_RETRY_PERIOD 1000 * 3          // If sync fails, try again after this
#define NTP_PORT 123                      // Used for Network Time Protocol

// Network
#define DHCP_SNIFF_INT 1000 * 30 // Search for DHCP every 30 secs
#define MAX_IP_CLIENTS 3         // Max simultaneous connections
#define NET_PORT 8080            // Used by software to connect

uint8_t mac[6];
bool dhcp_enabled = false;
unsigned long last_dhcp_search = 0;

// Initialize the EthernetUdp server library
// which is used to synchronize time across devices
const char timeServer[] = "time.nist.gov";    // Specify a reliable NTP server
const char apipaServer[] = "169.254.255.255"; // Specify a local APIPA server
const int NTP_PACKET_SIZE = 48;               // NTP time stamp is in the first 48 bytes of the message
byte packetBuffer[NTP_PACKET_SIZE];           // buffer to hold incoming and outgoing packets
EthernetUDP Udp;                              // A UDP instance to let us send and receive packets over UDP
IPAddress SUBNET_MASK = IPAddress(255, 255, 0, 0);

// Time sync variables
unsigned long last_TX_TS = 0;     // cached millis() from last NTP time sync TX
unsigned long last_RX_TS = 0;     // cached millis() from last NTP time sync RX
unsigned long last_TOI_TS = 0;    // Time when last RX'd NTP packet was generated
unsigned long last_EPOCH_sec = 0; // NTP seconds (since 1970) at last_TOI_TS localtime
unsigned long last_EPOCH_ms = 0;  // NTP milliseconds (0-1000) at last_TOI_TS localtime
short drift_TS = 0;               // crystal offset for getSystemTime() over TS_SYNC_PERIOD
byte NTP_retries = 0;             // Counter for tracking NTP packet retries
bool NTP_pending = false;         // Flag for tracking NTP packet retries
int NTP_local_pkts = 0;           // Stat counter of local NTP packets heard
int NTP_remote_pkts = 0;          // Stat counter of remote NTP packets heard
bool NTP_local_master = true;     // This device is acting as the local NTP server

// Initialize the Ethernet server library
// with the local device port (DHCP assigned IP)
EthernetServer server(NET_PORT);
EthernetClient IP_clients[MAX_IP_CLIENTS];
int maxClients = 0;

#endif

/********************* VARIABLE DECLARATION ********************/

NvMem nv = NvMem();

byte LED_WHITE_PIN = LED_WHITE_PIN_1; // micro_led moved to pin 5, but this might conflict, so disable

// Create the AD9851 DDS/DAC and ADC interface objects
AD9851 ad9851 = AD9851();
AD8302 ad8302 = AD8302();

#if USE_L298NHB
// Create the L298NHB PWM temperature setpoint object
L298NHB l298nhb = L298NHB(L298NHB_M1, L298NHB_E1, L298NHB_M2, L298NHB_E2, L298NHB_COOL, L298NHB_INIT);
unsigned long l298nhb_task_timer = 0;  // time of most recent update
unsigned long l298nhb_auto_off_at = 0; // time to auto-off (if no input)
int8_t l298nhb_status = 0;
bool user_warned = false;
#endif

// Create the MCP9808 and/or MAX31855 temperature sensor object
#if USE_MAX31855
MAX31855 max31855 = MAX31855(MAX31855_CLK_1, MAX31855_CS, MAX31855_SO_1, MAX31855_WAIT);
#endif
MCP9808 mcp9808 = MCP9808(); // always define (shutdown if unused)
float temperature = NAN;
float ambient = NAN;

// HW-agnostic interface pointer:
// For TEENSY36: always the Serial port
// For TEENSY41: also EthernetClient(s)
Stream *client = &Serial;

#if USE_ILI9341
ILI9341_t3 tft = ILI9341_t3(TFT_CS, TFT_DC, TFT_RST, TFT_MOSI, TFT_SCLK, TFT_MISO);
#endif

IntervalTimer sleepTimer;
const unsigned long sleepTimerDelay_us = 1000 * 1000 * 60 * 15;
const unsigned long sleepTimerInt_us = 1000 * 1000 * 5;
unsigned long sleep_timer_delay = sleepTimerDelay_us; // overridden when "SLEEP" cmd used
unsigned long time_of_last_msg = 0;                   // idle time tracker for screensaver
bool sleep_running = false;
byte last_temp_label = 255; // none
char last_line_label[16];   // trailing NULL
float last_pv, last_sp, last_op;
short last_pct;

// double slf_val;
float max_mag_flt = 0;

// stream state variables
bool is_running = false;
bool check_dtr = false;
bool streaming = false;
bool identifying = false;
unsigned long stop_identify_at = 0;
unsigned long swSpeed_us = 0;

// averaging parameters (settable)
byte avg_in = 5;
byte avg_out = 5;
byte step_size = 6;
int max_drift_hz_l = 10000;
int max_drift_hz_r = 10000;
float power_point = 0.933;

// averaging parameters (calculated)
byte avg_in_hop = 1;
byte avg_out_hop = 1;
float smooth_factor_out_hop = -1; // flag to calculate at first-run
float smooth_factor_in_hop = -1;  // flag to calculate at first-run
float smooth_factor_out = 1;
float smooth_factor_in = 1;

// init sweep params
long freq_start;
long freq_stop;
long freq_base;
long freq_start_up;
long freq_stop_up;
long freq_base_up;
long freq_start_down;
long freq_stop_down;
long freq_base_down;
long base_overtones_per_cycle;
long base_overtone_counter;
byte overtone = 0;
bool base_increment = true;

// Flags for triggering and tracking serial I/O
bool message = false;
unsigned long pre_time = 0;
unsigned long last_time = 0;
unsigned long last_temp = 0;
unsigned long stream_start = 0;

// Retained for calibration sweeps
int targetDelta_mag = 0;
int lastLastVal_mag = 0;

// Timeout timer to prevent it from streaming on forever
long quack_counter = 0;
long quack_interval = 1000000; // 1m samples is ~20 minutes @ 1.2ms/s

// sweep loop
long n = 0; // sequence counter
long app_freq = 0;
float avg_mag = 0;
int dir = -1;
long peak_freq_avg = 0;
long peak_freq_l = 0;
long peak_freq_h = 0;
long peak_freq_left = 0;
long peak_freq_right = 0;
int peak_mag_avg = 0;
int peak_mag_l = 0;
int peak_mag_h = 0;
unsigned long peak_time_avg = 0;
unsigned long peak_time_l = 0;
unsigned long peak_time_h = 0;
unsigned long peak_time = 0;

int dir_up = 1;
long peak_freq_l_up = 0;
long peak_freq_h_up = 0;
long peak_freq_left_up = 0;
long peak_freq_right_up = 0;
int peak_mag_l_up = 0;
int peak_mag_h_up = 0;
unsigned long peak_time_l_up = 0;
unsigned long peak_time_h_up = 0;

int dir_down = 1;
long peak_freq_l_down = 0;
long peak_freq_h_down = 0;
long peak_freq_left_down = 0;
long peak_freq_right_down = 0;
int peak_mag_l_down = 0;
int peak_mag_h_down = 0;
unsigned long peak_time_l_down = 0;
unsigned long peak_time_h_down = 0;

unsigned long db_desired = 0;
unsigned long db_actual = 0;

long freq_overshoot = -1, left, right;
bool new_sample = false;

// error detection flags
bool net_error = false;
bool hw_error = false;
bool tft_error = false;

bool tft_msgbox = false;
byte msgbox_icon = 0; // error, fail, pass
char msgbox_title[32] = "QATCH nanovisQ";
char msgbox_text[32] = "No message provided.";

byte EEPROM_pid = 0;

IntervalTimer busyTimer;
volatile byte busyTimerState;
const unsigned long busyTimerInt_us = 15000;

void busyTimerTask(bool dp)
{
  if (busyTimerState >= 10 * 6)
    busyTimerState = 0;
  byte seg = busyTimerState / 10;
  byte seq = busyTimerState % 10;

#if USE_ILI9341
  if (seq == 0)
  {
    String msg = dp ? "Booting." : "Programming.";
    digitalToggle(LED_SEGMENT_DP);

    if (!HW_REV_MATCH(HW_REVISION_0))
    {
      tft_wakeup();

      if (seg == 0)
      {
        tft.fillRect(0, 0, TFT_WIDTH, 20, ILI9341_WHITE);
        tft.drawFastHLine(0, 21, TFT_WIDTH, ILI9341_BLACK);
        tft.setCursor(3, 3);
        tft.print(msg);
      }
      else
      {
        uint16_t h, w;

        tft.measureChar('.', &w, &h);
        int startPos = msg.length() - 1;
        tft.setCursor(3 + (w * (startPos + seg)), 3); // measureChar() assumes fixed-width font
        tft.print(".");
      }
    }
  }
#endif

  busyTimerState++;
}

// // **********************************************************
// // Function to do a byte swap in a byte array
// void RevBytes(byte *arr, size_t len)
// {
//   for (uint i1 = 0; i1 < len / 2; i1++)
//   {
//     uint i2 = len - 1 - i1;
//     if (i1 == i2)
//       break; // if odd number of values, last one is the same index, skip it
//     byte t = arr[i1];
//     arr[i1] = arr[i2];
//     arr[i2] = t;
//   }
// }

// // Auxuliary function to print the array
// void printArray(byte arr[], int n)
// {
//   // Serial.print("n: ");
//   // Serial.println(n);
//   Serial.print("Array: ");
//   for (int i = 0; i < n; i++)
//   {
//     // Serial.print("i: ");
//     // Serial.println(i);
//     Serial.print(arr[i]);
//     Serial.print(" ");
//   }
//   Serial.println();
// }

void nv_init()
{
  // Initialize NVMEM object with persistent storage in EEPROM
  if (!nv.load())
  {
    Serial.println("Warning: NVMEM failed to load. Attempting to restore to defaults...");
    NVMEM = nv.defaults();
    // save and reload stats with defaults,
    // sets nv.isValid() true if successful
    nv.save(); // isValid() must be true for save to work, set by defaults() call
    if (!nv.load())
      Serial.println("FAILURE: NVMEM cannot be restored to defaults! EEPROM is failing.");
    else
      Serial.println("SUCCESS: NVMEM restored to defaults.");
  }

  if (nv.isValid())
  {
    // detect hw revision (if unknown)
    if (HW_REV_MATCH(HW_REVISION_X))
    {
      Serial.println("Detecting HW Revision...");
      if (detect_hw_revision()) // hw revision found
        nv.save();
    }
    config_hw_revision(NVMEM.HW_Revision);

    EEPROM_pid = NVMEM.pid;
    max31855.setOffsetA(byte_to_float(NVMEM.OffsetA));
    max31855.setOffsetM(byte_to_float(NVMEM.OffsetM));

    if (millis() > 3000)
      Serial.printf("NVMEM re-initialized @ %u ms!\n", millis());
  }
}

bool detect_hw_revision(void)
{
  byte HW_REVs[2] = {HW_REVISION_0, HW_REVISION_1};
  byte MAX_CLKs[2] = {MAX31855_CLK_0, MAX31855_CLK_1};
  byte MAX_SOs[2] = {MAX31855_SO_0, MAX31855_SO_1};
  for (unsigned int i = 0; i < (sizeof(MAX_CLKs) / sizeof(byte)); i++)
  {
    max31855 = MAX31855(MAX_CLKs[i], MAX31855_CS, MAX_SOs[i], MAX31855_WAIT);
    max31855.begin();
    /// @note detection requires valid internal and external probe readings
    if ((max31855.status() & ~MAX_FAULT_OPEN) != MAX_STATUS_OK) // not 'ok' (ignore 'open' external sensor fault)
    {
      continue; // try next HW rev
    }
    else
    {
      if (HW_REVs[i] == HW_REVISION_1)
      {
        // Initially Rev. 2 looks like Rev. 1 HW...
        // Check to see if this is really Rev. 2 HW
        // Test: Look for solder short on PIN 37/38
        bool pins_shorted = false;
        pinMode(AD8302_MAG36, INPUT_PULLDOWN);
        pinMode(AD8302_MAG41, OUTPUT);
        digitalWrite(AD8302_MAG41, HIGH);
        if (digitalRead(AD8302_MAG36)) // 37 pulldown, 38 set high (if high, likely short)
        {
          pinMode(AD8302_MAG36, INPUT_PULLUP);
          digitalToggle(AD8302_MAG41);
          if (!digitalRead(AD8302_MAG36)) // 37 pullup, 38 set low (if low now, definitely shorted)
            pins_shorted = true;
        }
        pinMode(AD8302_MAG36, INPUT);
        pinMode(AD8302_MAG41, INPUT);
        if (!pins_shorted)
          HW_REVs[i] = HW_REVISION_2; // upgrade Rev. 1 to Rev. 2 when solder short is not present
      }

      NVMEM.HW_Revision = HW_REVs[i];
      Serial.printf("Detected HW: Rev %u\n", HW_REVs[i]);
      return true; // works, we found our HW rev
    }
  }
  return false; // no HW rev detected
}

void config_hw_revision(byte hw_rev)
{
  Serial.printf("Configuring HW Rev: %u\n", hw_rev);
  switch (hw_rev)
  {
  case HW_REVISION_0:
    /// @note LED SEGMENT DISPLAY is not supported for Rev0 devices running new FW!
    /// For Rev0, only the DP segment will blink, no other segments are written to.
    max31855 = MAX31855(MAX31855_CLK_0, MAX31855_CS, MAX31855_SO_0, MAX31855_WAIT);
    LED_WHITE_PIN = LED_WHITE_PIN_0;
    break;

  case HW_REVISION_1:
  case HW_REVISION_2:
  case HW_REVISION_X: // if unknown, use most recent (assume new HW in HW error)
    max31855 = MAX31855(MAX31855_CLK_1, MAX31855_CS, MAX31855_SO_1, MAX31855_WAIT);
    LED_WHITE_PIN = LED_WHITE_PIN_1;
    break;
  }
  max31855.begin();   // re-initialize 'hw_detect' after re-instanciating object
  config_hw_ad9851(); // re-configure AD9851 'REFCLK' based on new 'HW_Revision'
}

void config_hw_ad9851(void)
{
  // Determine REFCLK frequency and multiplier (based on HW revision)
  unsigned long REFCLK_freq = ad9851.REFCLK_125MHz;
  bool REFCLK_6x_enable = false;

  if (HW_REV_MATCH(HW_REVISION_2)) // 30MHz multiplied up to 180MHz
  {
    REFCLK_freq = ad9851.REFCLK_30MHz; // AD9851 will 6x this internally
    REFCLK_6x_enable = true;
  }

  Serial.print("REFCLK: ");
  Serial.println(REFCLK_freq);
  Serial.print("6X REFCLK enable: ");
  Serial.println(REFCLK_6x_enable);

  // Initialize AD9851 and put to sleep
  ad9851.begin(DATA, WCLK, FQ_UD, REFCLK_freq, REFCLK_6x_enable);
  ad9851.shutdown();
  is_running = false;
}

float byte_to_float(byte b)
{
  if (b == 0xFF)
    b = 0; // handle special value (unset)
  float f;
  signed char c = b;
  if (c & 0x80)
    c = -(~c); // convert unsigned to signed (don't add 1)
  // convert raw offset byte to degrees C
  f = (c / CAL_FACTOR); // 0.05x resolution
  return f;
}

byte float_to_byte(float f)
{
  byte b;
  if (f < 0)
  {
    if (f < -6.35)
      f = -6.35;
    b = ~(byte)(-f * CAL_FACTOR); // don't add 1
  }
  else
  {
    if (f > 6.35)
      f = 6.35;
    b = (f * CAL_FACTOR);
  }
  return b;
}

/**************************** SETUP ****************************/

void QATCH_setup()
{
  // Initialize serial communication
  Serial.begin(BAUD);

  // Give a sign of life, then load NVMEM
  ledWrite(LED_SEGMENT_DP, HIGH);
  nv_init();

  // Show splash screen and perform TFT pixel read test
  tft_splash(true);

  busyTimerState = 0;
  busyTimer.begin([]
                  { busyTimerTask(true); },
                  busyTimerInt_us);

  // Seed random number generator with randomness
  int seed = micros();
  for (uint8_t pin = 0; pin < 42; pin++)
    seed += analogRead(pin);
  randomSeed(seed);

  // // Input array
  // // byte arr[] = {1, 2, 3, 4, 5};
  // // Size of the input array
  // // int n = sizeof(arr) / sizeof(arr[0]);

  // NvMem memory;
  // // memory.structure.field1 = 0;
  // // memory.structure.field2 = 6;
  // // String name = "DEAD,BEEF!";
  // memory.structure.field1 = 1.2345;
  // memory.structure.field2 = 9;
  // String name = "TEST,TEST!";
  // name.toCharArray(memory.structure.name, sizeof(memory.structure.name));

  // byte *arr = (byte *)&memory;
  // int n = sizeof(NvMem_RAM) / sizeof(byte);

  // // Print the array
  // printArray(arr, n);

  // // call the function for the reverse
  // RevBytes(arr, n);

  // Serial.println("Values written to EEPROM:");
  // // Print the array
  // printArray(arr, n);

  // // NvMem_RAM revmem;
  // // memcpy(&arr[0], &revmem, n);

  // // EEPROM.put(NVMEM_LENGTH - NVMEM_OFFSET - n - 1, revmem);
  // for (int i = 0; i < n; i++)
  // {
  //   int address = NVMEM_LENGTH - NVMEM_OFFSET - n - 1 + i;
  //   EEPROM.update(address, arr[i]);
  //   // Serial.print("Writing EEPROM address ");
  //   // Serial.print(address);
  //   // Serial.print(" with value ");
  //   // Serial.println(arr[i]);
  // }
  // for (int i = NVMEM_LENGTH - NVMEM_OFFSET - n - 1 - 1; i >= 0; i--)
  // {
  //   if (EEPROM.read(i) != 0xFF)
  //     EEPROM.write(i, 0xFF);
  //   else
  //   {
  //     Serial.print("Stopped scanning for empty EEPROM at address ");
  //     Serial.println(i);
  //     break;
  //   }
  // }

  // // NvMem_RAM reversed_memory;
  // // EEPROM.get(NVMEM_LENGTH - NVMEM_OFFSET - n - 1, reversed_memory);
  // // byte *readback = (byte *)&reversed_memory;
  // byte readback[n];
  // for (int i = 0; i < n; i++)
  // {
  //   int address = NVMEM_LENGTH - NVMEM_OFFSET - n - 1 + i;
  //   readback[i] = EEPROM.read(address);
  //   // Serial.print("Reading EEPROM address ");
  //   // Serial.print(address);
  //   // Serial.print(" with value ");
  //   // Serial.println(readback[i]);
  // }
  // printArray(readback, n);
  // RevBytes(readback, n);
  // printArray(readback, n);

  // memcpy(&memory, readback, n);

  // Serial.println(memory.structure.field1);
  // Serial.println(memory.structure.field2);
  // Serial.println(memory.structure.name);
  // Serial.println(memory.structure.field3[0]);
  // Serial.println(memory.structure.field3[1]);
  // Serial.println(memory.structure.field3[2]);
  // Serial.println(memory.structure.field3[3]);
  // Serial.println(memory.structure.field3[4]);

  if (tft_error)
    Serial.println("TFT ERROR: LCD may flash excessively on some UI redraws.");

  // Initialize I2C communication as Master
  Wire.begin();

  // set potentiometer value
  // Send I2C transmission to AD5252
  Wire.beginTransmission(POT_ADDRESS);
  // Send instruction for POT channel-0
  Wire.write(POT_CHANNEL);
  // Input resistance value, 0x80(128)
  Wire.write(POT_VALUE);
  // Stop I2C transmission
  Wire.endTransmission();

  // AD9851 is now initialized in 'config_hw_revision()'
  // See: 'config_hw_ad9851()' and do nothing here

  // Initialize AD8302 and configure ADC library
  ad8302.begin(AD8302_PHASE, AD8302_MAG36, AD8302_MAG41, AD8302_REF);
  ad8302.config(ADC_RESOLUTION, MAG_AVG, AD8302_MAG);

#if USE_L298NHB
  // Initialize L298NHB PWM for TEC
  // Nothing to do
#endif

#if !USE_ILI9341
  Serial.println("Using legacy hardware GPIOs for temperature sensor (USE_ILI9341 == false).");
#endif

#if USE_MAX31855
  if (PID_IS_SECONDARY(NVMEM.pid))
    Serial.printf("Skipping temperature hardware checks. PID=%u\n", EEPROM_pid);
  else
  {
    if (max31855.begin()) // detect HW type (MAX6675 vs MAX31855)
      Serial.println("Using MAX6675 sensor for temperature.");
    else
      Serial.println("Using MAX31855 sensor for temperature.");

    if (max31855.status() != 0) // not 'OK'
    {
      Serial.println("HW Issue: Temperature processor not responding!");
      hw_error = true;
    }

    temperature = max31855.readCelsius(false);
    if (temperature == 0 || isnan(temperature)) // not 'OK'
    {
      Serial.println("HW Issue: External temperature sensor not responding!");
      hw_error = true;
    }

    ambient = max31855.readInternal(false);
    if (ambient == 0 || isnan(ambient)) // not 'OK'
    {
      Serial.println("HW Issue: Internal temperature sensor not responding!");
      hw_error = true;
    }
  }
#endif

#if USE_MCP9808
  // begin MCP9808 temperature sensor
  bool mcp_ready = mcp9808.begin(); // start I2C interface
  mcp9808.shutdown();               // shutdown MSP9808 - power consumption ~0.1 mikro Ampere, stops temperature sampling

  hw_error |= !mcp_ready; // only add-in error if used
  Serial.println("Using MCP9808 sensor for temperature.");

  if (hw_error)
    Serial.println("HW Issue: Temp sensor not responding!");
#endif

  // Turn on LEDs at boot
  byte _led_pwr = 32;
  ledWrite(LED_RED_PIN, _led_pwr);
  ledWrite(LED_BLUE_PIN, _led_pwr);
  ledWrite(LED_ORANGE_PIN, _led_pwr);
  ledWrite(LED_WHITE_PIN, _led_pwr);

  // Set FAN on at boot
  pinMode(TEMP_CIRCUIT, OUTPUT);
  pinMode(TEC_SENSE, OUTPUT);
  digitalWrite(TEMP_CIRCUIT, HIGH);
  digitalWrite(TEC_SENSE, HIGH);

  /// @note Set by NVMEM in nv_init() now
  // Pull PID from EEPROM to RAM
  // EEPROM_read_pid();

#if HW_MATCH(TEENSY36)
  {
    delay(100); // no Ethernet init for 3.6 HW (just blink LED)
  }
#elif HW_MATCH(TEENSY41)
  {
    // Read HW-IDs from chip
    teensyMAC(mac);

    // attempt Ethernet configure
    connect_to_ethernet();
  }
#else // if HW_MATCH (HW_OTHER)
  {
    Serial.println("FATAL: UNRECOGNIZED HW");
    while (1)
      ; // stop here, unrecognized hardware
  }
#endif

  // Turn off LEDs and FAN after boot check
  if (hw_error)
    ledWrite(LED_ORANGE_PIN, HIGH);
  else
    ledWrite(LED_ORANGE_PIN, LOW);
  //  if (net_error) ledWrite(LED_ORANGE_PIN, HIGH);
  //  else ledWrite(LED_ORANGE_PIN, LOW);
  ledWrite(LED_RED_PIN, LOW);
  ledWrite(LED_BLUE_PIN, LOW);
  ledWrite(LED_WHITE_PIN, LOW);
  digitalWrite(TEMP_CIRCUIT, LOW);
  digitalWrite(TEC_SENSE, LOW);

  // Initialize smoothing factors
  smooth_factor_out = toSmoothFactor(avg_out);
  smooth_factor_in = toSmoothFactor(avg_in);

  busyTimer.end();
  digitalWrite(LED_SEGMENT_DP, LOW);

  //  tft_testmode();
  //  tft_tempcontrol();
  // tft_progress_test();
  tft_idle();

  // Serial.print("EEPROM length: ");
  // Serial.println(EEPROM.length());
}

// helper function since PJRC library doesn't handle mode switching
void ledWrite(int pin, int value)
{
  if (pin < 255) // valid pin
  {
    // NOTE: Analog value 1 will be interpreted as digital HIGH (DO NOT USE)
    pinMode(pin, OUTPUT); // required to switch modes: digital <-> analog
    if (value == HIGH || value == LOW)
      digitalWrite(pin, value);
    else
      analogWrite(pin, value);
  }
}

/**************************** LOOP *****************************/

void QATCH_loop()
{
  int bytesAtPort = Serial.available();
  int bytesAtClient = 0; // NONE (0)

#if HW_MATCH(TEENSY41)
  {
    EthernetClient newClient = server.accept();

    if (newClient) // new client
    {
#if defined NET_TCP_DEBUG
      Serial.println("[NET_TCP] found new client!");
#endif
      for (byte i = 0; i < MAX_IP_CLIENTS; i++)
      {
        if (!IP_clients[i])
        {
          // Once we "accept", the client is no longer tracked by EthernetServer
          // so we must store it into our list of IP_clients
          IP_clients[i] = newClient;

          IP_clients[i].println("HTTP/1.1 200 OK");
          IP_clients[i].println("Content-Type: text/plain");
          IP_clients[i].println("Connection: close"); // the connection will be closed after completion of the response
          IP_clients[i].println();

          int waitForReq_ms = 3000;
          while (!IP_clients[i].available() && --waitForReq_ms)
            delay(1); // wait for request before checking for available() bytes

#if defined NET_TCP_DEBUG
          if (!waitForReq_ms)
            Serial.println("[NET_TCP] Connection timeout! Client took too long to issue a GET request.");
          Serial.print("[NET_TCP] GET request took ");
          Serial.print(3000 - waitForReq_ms);
          Serial.println("ms");
#endif
          if (maxClients < i + 1)
            maxClients = i + 1;
          if (maxClients >= MAX_IP_CLIENTS)
            Serial.println("WARN: Too many clients. IP stack bug!");

#if defined NET_TCP_DEBUG
          Serial.printf("[NET_TCP] MAX CLIENTS: %u\n", maxClients);
          Serial.printf("[NET_TCP] ACT CLIENTS: %u\n", i + 1);
#endif
          break;
        }
      }
    }

    for (byte i = 0; i < MAX_IP_CLIENTS; i++)
    {
      if (IP_clients[i])
      {
        if (!IP_clients[i].connected())
        {
#if defined NET_TCP_DEBUG
          Serial.print("[NET_TCP] disconnected client");
          Serial.println(i);
#endif
          // Stop any clients which disconnect
          IP_clients[i].stop();
          client = &Serial; // talk back over Serial

          // Stop stream, if active
          if (streaming)
          {
            stopStreaming();
          }
        }
        else if (IP_clients[i].available())
        {
          // Flag first active client with incoming data
          // service oldest clients first (FIFO)
          if (!bytesAtClient)
          {
#if defined NET_TCP_DEBUG
            Serial.print("[NET_TCP] bytes at client");
            Serial.println(i);
#endif
            bytesAtClient = i + 1; // idx offset 1
          }
        }
        else if (!(message || streaming))
        {
#if defined NET_TCP_DEBUG
          Serial.print("[NET_TCP] stopping client");
          Serial.println(i);
#endif
          // Client has nothing to left to say and device is not streaming
          // so stop the client to end the session
          delay(1); // wait for client write to flush!
          IP_clients[i].stop();
          client = &Serial; // talk back over Serial
        }
      }
    }

    // Check for Ethernet cable present
    if (Ethernet.linkStatus() == LinkON)
    {
      // Clear error LED
      //      if (!hw_error) ledWrite(LED_RED_PIN, LOW);
      //      ledWrite(LED_ORANGE_PIN, LOW);

      // skip these tasks if streaming, to focus on speed:
      if (!streaming)
      {
        // Reconnect to Ethernet on network link change
        if (net_error || (!dhcp_enabled && millis() - last_dhcp_search > DHCP_SNIFF_INT))
        {
          dhcp_enabled = search_for_dhcp();
          net_error = false;
          // resync time on phy change if DHCP enabled and no prior sync
          if (dhcp_enabled && last_TOI_TS == 0)
          {
            last_RX_TS = 0; // do sync now
          }
        }

        // Handle refreshing stale a time sync
        // by sending, receiving and processing NTP packets
        switch (checkForTimeSyncUpdates())
        {
        case 0:
          // nothing happened
          break;
        case 1:
          Serial.println("NTP packet sent");
          break;
        case 2:
          Serial.println("NTP packet received");
          break;
        case 3:
          Serial.println("NTP packet retry");
          break;
        case 4:
          Serial.println("NTP packet timeout");
          break;
        default:
          Serial.println("NTP timesync error");
          break;
        }

        // Maintain DHCP lease (if needed)
        switch (Ethernet.maintain())
        {
        case 0:
          // nothing happened
          break;
        case 1:
          Serial.println("IP renew failed");
          break;
        case 2:
          Serial.println("IP renew success");
          break;
        case 3:
          Serial.println("IP rebind fail");
          break;
        case 4:
          Serial.println("IP rebind success");
          break;
        default:
          Serial.println("IP maintain error");
          break;
        }
      }
    }
    else // LinkOFF
    {
      if (!net_error)
      {
        Ethernet.setLocalIP(IPAddress(0, 0, 0, 0));
        net_error = true;
        dhcp_enabled = false;
      }

      //      // Blink error LED
      //      unsigned short mod_mode = getSystemTime() % 3000;
      //      if (mod_mode > 800 && mod_mode < 2300)
      //      {
      //        //        if (!check_dtr) ledWrite(LED_RED_PIN, HIGH);
      //        ledWrite(LED_ORANGE_PIN, HIGH);
      //      }
      //      else
      //      {
      //        //        if (!hw_error) ledWrite(LED_RED_PIN, LOW);
      //        ledWrite(LED_ORANGE_PIN, LOW);
      //      }

      // Indicate offline and out-of-sync
      if (last_TX_TS != 0 && isTimeForSync())
      {
        last_TX_TS = 0; // only do this once
        Serial.println("NTP timesync pending");
      }
    }
  }
#endif

  if (bytesAtPort || bytesAtClient)
  {
    //    Serial.println("Message received!"); // debug
    time_of_last_msg = micros();
    if (time_of_last_msg < sleepTimerInt_us && !sleep_running)
    {
      // this will prevent our world view from becoming misaligned with reality
      // (i.e. we could end sleep prematurely due to rollover, not a message rx)
      time_of_last_msg = sleepTimerInt_us;
    }
    if ((l298nhb_auto_off_at - millis()) < L298NHB_AUTOOFF && l298nhb.active()) // not yet time to auto off
    {
      l298nhb_auto_off_at = millis() + L298NHB_AUTOOFF; // calculate time to auto off
      // special case in event of rollover in prior line math
      if (l298nhb_auto_off_at == 0) 
      {
        l298nhb_auto_off_at = 1;
      }
    }

    String message_str;
    if (bytesAtPort)
    {

#if HW_MATCH(TEENSY41)
      // if switching from an IP client to serial port, and
      // streaming on IP client, stop and disconnect first
      if (client != &Serial)
      {
        // Stop stream, if active
        if (streaming)
        {
          stopStreaming();
          client->println("STOP");
          delay(1); // wait for client write to flush!
          ((EthernetClient *)client)->stop();
        }
      }
#endif

      // read message at serial port
      client = &Serial; // talk back over Serial
      message_str = Serial.readStringUntil('\n');
      // if (!message_str.endsWith('\n'))
      // {
      //   Serial.println("Buffering!");
      //   message_str += Serial.readStringUntil('\n');
      // }
    }

#if HW_MATCH(TEENSY41)
    else if (bytesAtClient)
    {
      // Stop stream, if active (on either interface)
      if (streaming)
      {
        stopStreaming();

        // if switching from one IP client to another (but not from Serial)
        // send "STOP" and disconnect from other IP client first
        if (client != &Serial)
        {
          client->println("STOP");
          delay(1); // wait for client write to flush!
          ((EthernetClient *)client)->stop();
        }
      }

      // read message at ethernet port
      client = &IP_clients[bytesAtClient - 1]; // talk back over IP
      message_str = client->readStringUntil('\n');

      int len;
      if ((len = message_str.indexOf('\n')))
      {
        message_str = message_str.substring(0, len);
        while (client->available())
        {
          if (client->peek() == ':')
            break;                       // stop if 'intel hex'
          client->readStringUntil('\n'); // toss header line
        }
      }
      // Serial.println(message_str);
      message_str = message_str.substring(message_str.indexOf(' ') + 1);
      message_str = message_str.substring(0, message_str.indexOf(' '));
      message_str = message_str
                        .replace('/', "")   // ignore root path specifier
                        .replace('&', '\n') // take first parameter only
                        .replace('=', ' ')  // convert equals to space
                        .replace("_", '?'); // convert underscore to question mark
      message_str = message_str.substring(message_str.indexOf('?') + 1);
      message_str = message_str.substring(0, message_str.indexOf('\n'));

      if (message_str.toLowerCase() != "favicon.ico")
      {
        Serial.print("[IP CMD] Received: \"");
        Serial.print(message_str);
        Serial.println("\"");
      }
      else // favicon request
      {
        // ignore it, we'll close it immediately (on next loop)
      }
    }
#endif

    char buf[message_str.length() + 1]; // trailing NULL
    message_str.toCharArray(buf, sizeof(buf));
    char *p = buf;
    char *str;
    int nn = 0;

    quack_counter = 0; // reset stream timeout counter

    if (message_str.toUpperCase() == "VERSION")
    {
      client->println(DEVICE_BUILD);
      client->println(CODE_VERSION);
      client->println(RELEASE_DATE);
      return;
    }

    if (message_str.toUpperCase() == "INFO")
    {
      // Added for v2.3x in v2.2s
      switch (BOARD_TYPE)
      {
      case TEENSY36:
        client->println("HW: TEENSY36");
        break;
      case TEENSY41:
        client->println("HW: TEENSY41");
        break;
      case HW_OTHER:
        client->println("HW: ???");
        break;
      }
#if HW_MATCH(TEENSY41)
      client->print("IP: ");
      client->println(Ethernet.localIP());
      client->print("MASK: ");
      client->println(Ethernet.subnetMask());
      client->print("GATEWAY: ");
      client->println(Ethernet.gatewayIP());
      client->print("DHCP: ");
      client->print(Ethernet.dhcpServerIP());
      client->printf(" (%s)\n", dhcp_enabled ? "ON" : "OFF");
      client->print("DNS: ");
      client->println(Ethernet.dnsServerIP());
      client->printf("UID: %s\n", teensyUID64());
#endif
      client->printf("MAC: %s\n", teensyMAC());
      client->printf("USB: %u\n", teensyUsbSN());
      client->printf("PID: %X\n", EEPROM_pid); // Position ID
      // PID is programmed using serial CMD: "EEPROM {addr} {val}" where addr=0
      client->printf("REV: %u\n", NVMEM.HW_Revision);

      client->print("ERR: ");
      bool ignore_temp_errors = PID_IS_SECONDARY(NVMEM.pid);
      if (ignore_temp_errors)
        hw_error = false; // prevent temp sensor errors on secondary multiplex devices
      else
        hw_error = (max31855.status() != 0);
      if (hw_error)
      {
        client->println("TEMP SENSOR " + max31855.status(true) + " (" + max31855.status(false) + ")");
      }
      else if (tft_error)
      {
        client->println("TFT TEST FAILED [SCREEN MAY FLASH]");
      }
      else if (max31855.error() != "OK[NONE]" && !ignore_temp_errors)
      {
        client->println("TEMP SENSOR [TRANSIENT] " + max31855.error());
      }
      else
      {
        client->println("NONE");
      }
      return;
    }

    if (message_str.startsWith("MULTI"))
    {
      if (message_str.substring(6, 10) == "INIT")
      { // support: start, begin, 1, stop, end, 0
        if (message_str.endsWith("START") ||
            message_str.endsWith("BEGIN") ||
            message_str.endsWith("1"))
        {
          tft_initialize();
          client->println("1"); // started
        }
        else if (message_str.endsWith("STOP") ||
                 message_str.endsWith("END") ||
                 message_str.endsWith("0"))
        {
          tft_idle();
          client->println("0"); // stopped
        }
        else
        {
          client->println("?"); // unknown
        }
      }
      else
      {
        client->println("?"); // unknown
      }
      return;
    }

    if (message_str.startsWith("MSGBOX"))
    {
      // ex) "MSGBOX PASS:INITIALIZE SUCCESS;READY TO MEASURE"
      if (message_str.endsWith("MSGBOX"))
      {
        tft_msgbox = false;
        client->println("0"); // cleared
      }
      else
      {
        if (message_str.indexOf(";") > 0)
        {
          String msg_cmd = message_str.substring(7);
          if (msg_cmd.indexOf(":"))
          {
            if (msg_cmd.startsWith("ERROR"))
              msgbox_icon = 0; // error
            else if (msg_cmd.startsWith("FAIL"))
              msgbox_icon = 1; // fail
            else if (msg_cmd.startsWith("PASS"))
              msgbox_icon = 2; // pass
            else
              msgbox_icon = 0; // error
            msg_cmd = msg_cmd.substring(msg_cmd.indexOf(":") + 1);
          }
          byte sep_idx = msg_cmd.indexOf(";");
          msg_cmd.substring(0, sep_idx).toCharArray(msgbox_title, sizeof(msgbox_title));
          msg_cmd.substring(sep_idx + 1).toCharArray(msgbox_text, sizeof(msgbox_text));
          tft_msgbox = true;
          client->println("1"); // updated
        }
        else
        {
          client->println("?"); // unknown
        }
      }
      tft_idle(); // redraw msgbox
      return;
    }

#if HW_MATCH(TEENSY41)
    if (message_str.startsWith("SYNC"))
    {
      // no SYNC parameter given
      // or SYNC NOW
      if (message_str.endsWith("SYNC") ||
          message_str.endsWith("NOW"))
      {
        // do sync now
        last_RX_TS = 0;
      }

      // SYNC STATUS
      if (message_str.endsWith("STATUS"))
      {
        unsigned long nowEpoch = getNowEpoch();
        long secsSinceLastSync = nowEpoch - last_EPOCH_sec;
        short syncState = getSyncState();

        client->printf("NTP STATE:  %s (%i)\n", syncState < 0 ? "NO-SYNC" : syncState > 0 ? "OUT-OF-SYNC"
                                                                                          : "IN-SYNC",
                       syncState);
        client->printf("NTP ROLE:   %s\n", NTP_local_master ? "MASTER" : "SLAVE");
        client->printf("NTP HITS:   %u / %u\n", NTP_local_pkts, NTP_remote_pkts);
        client->printf("LAST TX:    %u\n", last_TX_TS);
        client->printf("LAST RX:    %u\n", last_TOI_TS); // report TOI as RX so timeouts don't show
        client->printf("TIME ALIVE: %u\n", millis());
        client->printf("LAST SYNC:  %u (%i seconds)\n", last_EPOCH_sec, -secsSinceLastSync);
        client->printf("NOW EPOCH:  %u\n", nowEpoch);
        client->printf("NEXT SYNC:  %u (%i seconds)\n", (TS_SYNC_PERIOD / 1000) + last_EPOCH_sec, (TS_SYNC_PERIOD / 1000) - secsSinceLastSync);
        client->printf("SYS. TIME:  %u\n", getSystemTime());
        client->printf("LAST DRIFT: %i\n", drift_TS); // this must be printed as a signed value
        getSystemTime(true);                          // reports NOW DRIFT
      }
      return;
    }
#endif

    if (message_str.startsWith("TEMP"))
    {
      //      if (message_str.substring(5) == "FAIL")
      //      {
      //        max31855.simulate_err = !max31855.simulate_err;
      //        client->print("simulate_err: ");
      //        client->println(max31855.simulate_err);
      //        return;
      //      }
      if ((l298nhb_auto_off_at - millis()) < L298NHB_AUTOOFF && l298nhb.active()) // not yet time to auto off
      {
        l298nhb_auto_off_at = millis() + L298NHB_AUTOOFF; // calculate time to auto off
        // special case in event of rollover in prior line math
        if (l298nhb_auto_off_at == 0) 
        {
          l298nhb_auto_off_at = 1;
        }
      }
      bool updateAmbient = false;
      if (message_str.substring(5) == "ON")
      {
#if USE_L298NHB
        if (!l298nhb.active())
          updateAmbient = true;
        l298nhb.wakeup();
#endif
        digitalWrite(TEMP_CIRCUIT, HIGH);  // turn on fan
        digitalWrite(TEC_SENSE, LOW);  // low speed
      }
      else if (message_str.substring(5) == "OFF")
      {
#if USE_L298NHB
        if (l298nhb.active())
        {
          l298nhb.shutdown();
          // digitalWrite(TEMP_CIRCUIT, LOW); // turn off fan at end of cooldown, not now
          // digitalWrite(FAN_HIGH_LOW, LOW);
          //        tft_tempcontrol(); // draw "OFF" state
          tft_cooldown_start(); // change status to "cooldown" and start countdown
        }
#else
        digitalWrite(TEMP_CIRCUIT, LOW); // turn off fan
        digitalWrite(TEC_SENSE, LOW);
#endif
      }
#if USE_MAX31855
      else if (message_str.substring(6, 12) == "OFFSET")
      {
        // cmd format: "TEMP _OFFSET 3.0" for offsetA
        //             "TEMP =OFFSET 3.0" for offsetB
        //             "TEMP -OFFSET 3.0" for offsetC
        //             "TEMP +OFFSET 3.0" for offsetH
        float t = strtof(&buf[13], NULL);
        if (message_str.substring(5, 6) == "_")
        {
          max31855.setOffsetA(t);
          // also update NVMEM and save it to EEPROM
          NVMEM.OffsetA = float_to_byte(t);
          if (nv.isValid())
            nv.save();
          else
            client->println("ERROR: Failed to save OffsetA in EEPROM. NVMEM struct is invalid.");
        }
        if (message_str.substring(5, 6) == "=")
        {
          max31855.setOffsetB(t);
        }
        if (message_str.substring(5, 6) == "-")
        {
          max31855.setOffsetC(t);
        }
        if (message_str.substring(5, 6) == "+")
        {
          max31855.setOffsetH(t);
        }
        if (message_str.substring(5, 6) == "~")
        {
          max31855.setOffsetM(t);
          // also update NVMEM and save it to EEPROM
          NVMEM.OffsetM = float_to_byte(t);
          if (nv.isValid())
            nv.save();
          else
            client->println("ERROR: Failed to save OffsetM in EEPROM. NVMEM struct is invalid.");
        }
      }
#endif
#if USE_L298NHB
      else if (message_str.substring(5, 9) == "TUNE")
      {
        const char *pid[6]; // an array of pointers to the pieces of the above array after strtok()
        char *ptr = NULL;
        byte idx = 0;
        byte num_items = 0;
        ptr = strtok(&buf[10], ","); // delimiter
        while (ptr != NULL)
        {
          pid[idx] = ptr;
          idx++;
          ptr = strtok(NULL, ",");
          if (idx >= 6)
            break;
        }
        num_items = idx;
        while (idx <= 5) // fill any unprovided values with provided values
        {
          pid[idx] = pid[idx % num_items];
          idx++;
        }
        if (DEBUG)
        {
          client->println("The Pieces separated by strtok():");
          for (int n = 0; n < idx; n++)
          {
            client->print(n);
            client->print("->");
            client->println(pid[n]);
          }
        }
        l298nhb.setTuning(
            atof(pid[0]), atof(pid[1]), atof(pid[2]),  // cool PIDs
            atof(pid[3]), atof(pid[4]), atof(pid[5])); // heat PIDs
      }
#endif
#if USE_L298NHB
      else if (message_str.substring(5) == "HEAT")
      {
        l298nhb.setSignal(L298NHB_HEAT);
      }
#endif
#if USE_L298NHB
      else if (message_str.substring(5) == "COOL")
      {
        l298nhb.setSignal(L298NHB_COOL);
      }
#endif
#if USE_MAX31855
      else if (message_str.substring(5, 10) == "CHECK")
      {
        int max_mag;
        long max_freq, dist = 1000000, f = 15000000;
        float t_start = max31855.readCelsius(), t_check = 0;
        bool check_pass = false, do_fast = false;
        if (message_str.substring(10).trim().length())
        {
          f = strtoul(&buf[11], NULL, 0);
          do_fast = true;
        }
        ad9851.wakeup();
        max_freq = f;
        for (int i = 1; i <= 1000; i *= 10)
        {
          freq_start = max_freq + (dist / i);
          freq_stop = freq_start - 2 * (dist / i);
          freq_base = dist / (100 * i);
          client->printf("CHECK PASS#%1.0f: %u,%u,%u,%u\n", log10(i) + 1, max_freq, freq_start, freq_stop, freq_base);
          if (do_fast && i < 100)
            continue;
          max_mag = max_freq = 0;
          app_freq = freq_start;
          ad9851.setFreq(app_freq);
          waitForAdcToSettle();
          while (app_freq > freq_stop)
          {
            app_freq -= freq_base;
            ad9851.setFreq(app_freq);
            waitForAdcToSettle();
            int app_mag = ad8302.read();
            if (app_mag > max_mag)
            {
              max_mag = app_mag;
              max_freq = app_freq;
            }
          }
        }
        ad9851.setFreq(max_freq);
        delay(200); // delay for temp to update
        max_mag = ad8302.read();
        t_check = max31855.readCelsius();
        check_pass = abs(t_start - t_check) < 1.0;
        if (PID_IS_SECONDARY(NVMEM.pid))
          check_pass = true;
        ad9851.shutdown();
        // report temp check results (detect sensor fault)
        client->printf("CHECK FREQ/MAG: %u,%u\n", max_freq, max_mag);
        client->printf("CHECK TEMPS: %2.2f,%2.2f\n", t_start, t_check);
        client->print("CHECK RESULT: ");
        client->println(check_pass ? "PASS" : "FAIL");
        return;
      }
#endif
      else if (message_str.substring(4).trim().length())
      {
        l298nhb_auto_off_at = millis() + L298NHB_AUTOOFF; // calculate time to auto off
        // special case in event of rollover in prior line math
        if (l298nhb_auto_off_at == 0) 
        {
          l298nhb_auto_off_at = 1;
        }
        if (!l298nhb.active())
          updateAmbient = true;

        // set DAC/PWM intensity
        float t;
        if (message_str.indexOf(".") < 0)
        {
          t = strtoul(&buf[5], NULL, 0);
        }
        else
        {
          t = strtof(&buf[5], NULL);
        }
#if USE_L298NHB
        if (l298nhb.getTarget() != t)
        {
          l298nhb.setTarget(t);
        }
#endif
      }

      if (updateAmbient)
      {
        l298nhb.wakeup();
#if USE_MCP9808
        mcp9808.wakeup();
        l298nhb.setAmbient(mcp9808.readTempC());
        mcp9808.shutdown();
#endif
#if USE_MAX31855
        // AJR TODO 2023-09-12: Disabled for old PID controller code
        // l298nhb.setAmbient(max31855.readCelsius());
        l298nhb.setAmbient(21.0); // always 21C, was: setAmbient(max31855.readInternal());
#endif
      }

      // write output to console
#if USE_L298NHB
      client->printf("L298NHB STATUS: ");
      int ext_5v_adc = analogRead(PIN_EXT_5V_VOLTAGE);
      float ext_5v_volts = L298NHB_VOLTAGE_CONVERT(ext_5v_adc);
      // warn user on first attempt to start
      bool v_err = false;
      if (!L298NHB_VOLTAGE_VALID(ext_5v_volts))
      {
        if (updateAmbient && l298nhb.getPower() == 1)
        {
          if (!user_warned)
          {
            l298nhb.setPower(0); // prevent temp circuit on, but retain setTarget()
            v_err = true;
            user_warned = true;

            // time to auto-off (forced)
            l298nhb_auto_off_at = 0;         // indicate not running, fan off if power issue
            digitalWrite(TEMP_CIRCUIT, LOW); // turn off fan
            digitalWrite(TEC_SENSE, LOW);
            ledWrite(LED_RED_PIN, LOW);
            ledWrite(LED_BLUE_PIN, LOW);
            tft_idle(); // revert back to idle, even if in cooldown, bc power issues
          }
        }
        else if (!l298nhb.active())
        {
          user_warned = false;
        }
      }
      if (l298nhb.active())
      {
        switch (l298nhb.getLabelState())
        {
        case 0:
          client->print("CYCLE,");
          break;
        case 1:
          client->print("WAIT,");
          break;
        case 2:
          client->print("CLOSE,");
          break;
        case 3:
          client->print("STABLE,");
          break;
        default:
          client->print("ERROR,");
          break;
        }
        if (l298nhb_status < 0)
          client->printf("COOL");
        if (l298nhb_status == 0)
          client->printf("IDLE");
        if (l298nhb_status > 0)
          client->printf("HEAT");
      }
      else if ((l298nhb_auto_off_at > 0) &&
               ((l298nhb_auto_off_at - millis()) > L298NHB_AUTOOFF)) // auto off triggered
      {
        client->printf("AUTO,OFF");
      }
      else if (v_err)
      {
        client->printf("ERROR,VOLTAGE");
      }
      else
      {
        client->printf("IDLE,OFF");
      }
      client->printf("\n");
      client->printf("L298NHB SETPOINT: %2.2f\n", l298nhb.getTarget());
      client->printf("L298NHB POWER: %u\n", l298nhb.getPower());
      client->printf("L298NHB VOLTAGE: %2.2fV (%u)\n", ext_5v_volts, ext_5v_adc);
      client->printf("TIME STABLE/TOTAL: %1.0f,%1.0f\n", l298nhb.timeStable(), l298nhb.timeElapsed());
      client->printf("TEMP MIN/MAX: %2.2f,%2.2f\n", l298nhb.getMinTemp(), l298nhb.getMaxTemp());
      l298nhb.resetMinMax();
#endif
#if USE_MCP9808
      mcp9808.wakeup();
      client->printf("MCP9808 TEMP: %2.2f\n", mcp9808.readTempC());
      mcp9808.shutdown();
#endif
#if USE_MAX31855
      if (isnan(temperature) || (millis() - last_temp > 2000))
        temperature = max31855.readCelsius();
      if (max31855.getType())
        client->print("MAX6675 STATUS: ");
      else
        client->print("MAX31855 STATUS: ");
      client->print(max31855.status(true)); // verbose status string
      client->print(" (");
      client->print(max31855.status(false)); // debug raw value
      client->println(")");

      if (max31855.getType())
      {
        mcp9808.wakeup(); // wake with no delay
        float t = mcp9808.readTempC();
        l298nhb.setAmbient(t);
        if (t == 0)
        {
          l298nhb.setAmbient(21.0);
        }
        client->printf("MCP9808 TEMP: %2.2f\n", t);
        mcp9808.shutdown();
      }
      else
      {
        float t = max31855.readInternal(false);
        // AJR TODO 2023-09-12: Disabled for old PID controller code
        // l298nhb.setAmbient(t);
        if (t == NAN)
        {
          l298nhb.setAmbient(21.0);
        }
        client->printf("INTERNAL TEMP: %2.2f\n", t);
      }
      client->printf("EXTERNAL TEMP: %2.2f\n", round(temperature / TEMP_RESOLUTION) * TEMP_RESOLUTION); // https://arduino.stackexchange.com/a/28469
      client->printf("EXTERNAL OFFSETS: %2.2f,%2.2f,%2.2f,%2.2f,%2.2f\n",
                     max31855.getOffsetA(),  // FW fixed offset (always)
                     max31855.getOffsetB(),  // SW defined (both)
                     max31855.getOffsetC(),  // SW defined (cool)
                     max31855.getOffsetH(),  // SW defined (heat)
                     max31855.getOffsetM()); // FW fixed offset (measure)
#endif
#if USE_L298NHB
      client->printf("PID TUNING: %.3g,%.3g,%.3g,%.3g,%.3g,%.3g\n",
                     l298nhb.getTuning(0),  // kpc
                     l298nhb.getTuning(1),  // kic
                     l298nhb.getTuning(2),  // kdc
                     l298nhb.getTuning(3),  // kph
                     l298nhb.getTuning(4),  // kih
                     l298nhb.getTuning(5)); // kdh
#endif

      return;
    }

    if (message_str.startsWith("EEPROM"))
    {
      /// @note To restore NVMEM to defaults, use command 'EEPROM -1 0xFF' to erase NVMEM.version byte and force a reset
      /// @note To set PID, use command 'EEPROM 0 [PID]' to set NVMEM.pid
      /// @note To set OffsetA (in encoded byte form), use command 'EEPROM 1 [OFFSET_BYTE]' to set NVMEM.OffsetA

      // read/write EEPROM data
      int address = strtoul(&buf[7], NULL, 0);
#if NVMEM_INVERT
      int new_address = NVMEM_LENGTH - NVMEM_OFFSET - address - 2; // minus 2 due to EEPROM address 0 mapping to PID @ NVMEM address 1
#else
      int new_address = NVMEM_OFFSET + address;
#endif
      byte value;
      int idx = message_str.substring(7).indexOf(" ") + 1;
      if (idx > 0)
      {
        value = strtoul(&buf[idx + 7], NULL, 0);
        EEPROM.update(new_address, value); // only write if different

        // Call update handlers, if any, for this address:
        // if (address == 0)
        //   EEPROM_read_pid();
        // if (address == 1)
        //   max31855.EEPROM_read();

        /// @note only call nv_init() if 'address' is in NVMEM struct region
        if (address < nv.size - 1) // minus 1 due to EEPROM address 0 mapping to PID @ NVMEM address 1
          nv_init();
      }
      else
      {
        value = EEPROM.read(new_address);
      }
      client->printf("EEPROM: %04x=%02x\n", address, value);
      return;
    }

    if (message_str.toUpperCase() == "PROGRAM")
    {
      // make sure LCD invert is OFF before programming
      if (identifying)
      {
        ledWrite(LED_WHITE_PIN, LOW);
        identifying = false;
        tft_identify(identifying);
      }

      // enter FlasherX bootloader
      tft_splash(false);
      busyTimerState = 0;
      busyTimer.begin([]
                      { busyTimerTask(false); },
                      busyTimerInt_us);
      flasher_update_task(client, LED_BLUE_PIN); // does not return
      return;
    }

    if (message_str.startsWith("SPEED"))
    {
      swSpeed_us = message_str.substring(6).toInt();
      // SW does not listen for a reply (should it?)
      // client->print("SPEED:");
      // client->println(swSpeed_us);
      return;
    }

    if (message_str.startsWith("AVG"))
    {
      // set new averaging parameters
      byte params = 0;
      while ((str = strtok_r(p, " ,;", &p)) != NULL) // delimiter is the semicolon
      {
        if (params == 1)
          avg_in = atol(str);
        if (params == 2)
          avg_out = atol(str);
        if (params == 3)
          step_size = atol(str);
        if (params == 4)
          max_drift_hz_l = atol(str);
        if (params == 5)
          max_drift_hz_r = atol(str);
        if (params == 6)
          power_point = atof(str); // float (not long)
        params++;
      }
      // read out parameters if none were provided
      if (params == 1)
      {
        client->printf("AVG_IN:         %u\n", avg_in);
        client->printf("AVG_OUT:        %u\n", avg_out);
        client->printf("STEP_SIZE:      %u\n", step_size);
        client->printf("MAX_DRIFT_HZ_L: %u\n", max_drift_hz_l);
        client->printf("MAX_DRIFT_HZ_R: %u\n", max_drift_hz_r);
        client->printf("POWER_POINT:    %0.3f\n", power_point);
      }
      // calculate new averaging parameters
      avg_in_hop = 1;             // set to default, calculate later (but only if hopping)
      avg_out_hop = 1;            // set to default, calculate later (but only if hopping)
      smooth_factor_out_hop = -1; // flag to calculate later, once base_overtones_per_cycle is known
      smooth_factor_in_hop = -1;  // flag to calculate later, once base_overtones_per_cycle is known
      smooth_factor_out = toSmoothFactor(avg_out);
      smooth_factor_in = toSmoothFactor(avg_in);
      return;
    }

    if (message_str.toUpperCase() == "STREAM")
    {
      tft_measure();
      streaming = true; // set streaming bit
      message = true;   // output messaging
      stream_start = micros();
      peak_time_avg = stream_start;
      peak_time_l = peak_time_h = stream_start;
      peak_time_l_up = peak_time_h_up = stream_start;
      peak_time_l_down = peak_time_h_down = stream_start;
      ledWrite(LED_WHITE_PIN, HIGH); // turn on light // TODO: make this adjustable brightness
      ledWrite(LED_SEGMENT_DP, HIGH);
      max31855.useOffsetM(true);
      return;
    }

    if (message_str.toUpperCase() == "STOP")
    {
      stopStreaming();
      client->println("STOP"); // SW listens for this reply
      return;
    }

    if (message_str.toUpperCase() == "SLEEP")
    {
      //      Serial.println("Sleep timer manually started"); // debug
      sleep_running = true;
      time_of_last_msg = 0;
      sleep_timer_delay = sleepTimerInt_us;
      sleepTimer.begin([]
                       { tft_screensaver(); },
                       sleepTimerInt_us);
      tft_screensaver(); // show immediately, don't wait for interval
      return;
    }

    if (message_str.toUpperCase() == "IDENTIFY")
    {
      identifying = !identifying;
      if (identifying)
      {
        stop_identify_at = millis() + 60000; // 1 min timeout
      }
      else
      {
        ledWrite(LED_WHITE_PIN, LOW);
      }
      tft_identify(identifying);
      client->println(identifying); // SW listens for this reply
      return;
    }

    // decode message
    byte params = 0;
    while ((str = strtok_r(p, " ,;", &p)) != NULL) // delimiter is the semicolon
    {
      params++;
      // frequency start
      if (nn == 0)
      {
        // only WAIT if freq_start is different than last time (hopping)
        freq_start = atol(str);
        // client->print("FREQ START = ");
        // client->println(freq_start);
        nn = 1;
      }
      // frequency stop
      else if (nn == 1)
      {
        freq_stop = atol(str);
        nn = 2;
      }
      // frequency step
      else if (nn == 2)
      {
        freq_base = atol(str);
        nn = 3;
        message = true; // mark sweep to begin!

        if (!streaming)
        {
          // reset global variables for new sweep
          swSpeed_us = 0;
          freq_start_up = freq_stop_up = freq_base_up = 0;
          freq_start_down = freq_stop_down = freq_base_down = 0;
          base_overtones_per_cycle = base_overtone_counter = 0;
        }
      }
      // frequency start up
      else if (nn == 3)
      {
        freq_start_up = atol(str);
        nn = 4;
      }
      // frequency stop up
      else if (nn == 4)
      {
        freq_stop_up = atol(str);
        nn = 5;
      }
      // frequency step up
      else if (nn == 5)
      {
        freq_base_up = atol(str);
        nn = 6;
      }
      // frequency start down
      else if (nn == 6)
      {
        freq_start_down = atol(str);
        nn = 7;
      }
      // frequency stop down
      else if (nn == 7)
      {
        freq_stop_down = atol(str);
        nn = 8;
      }
      // frequency step down
      else if (nn == 8)
      {
        freq_base_down = atol(str);
        nn = 9;
      }
      // base overtones per cycle
      else if (nn == 9)
      {
        base_overtones_per_cycle = atol(str);
        nn = 10; // ignore any additional delimited data!
      }
      else
      {
        // client->print("Ignored input: ");
        // client->println(nn);
      }
    }
    // process any additional messages
    // before going into action loop
    return;
  }

  // delayMicroseconds(1000000);

  if (message)
  {
    // start sweep
    long this_freq_start = freq_start;
    long this_freq_stop = freq_stop;
    long this_freq_base = freq_base;

    // int last_freq = 0;
    float last_mag = 0;
    float last_mag_1 = 0;
    float last_mag_2 = 0;
    int max_mag = 0;
    bool peak = false;
    long k = 0;
    byte samples_since_max = 0;
    byte num_max_samples = 0;
    long max_freq = 0;

    quack_counter++; // increment counter every sweep

    // stop stream if one-shot or "quacking" too long
    if (quack_counter > quack_interval ||
        !(DO_STREAM && streaming) ||
        (check_dtr && !Serial.dtr()))
    {
      stopStreaming();
    }

    if (!is_running)
    {
      // Update temperature now, if stale, prior to AD9851 wakeup (noisy)
      bool updateTemp = true; // ((millis() - last_temp) > 2000);
      if (updateTemp)
      {
        last_temp = millis();

        // measure temperature
#if USE_MCP9808
        temperature = mcp9808.readTempC();
#endif
#if USE_MAX31855
        if (isnan(temperature))
          temperature = max31855.readCelsius();
        else
          temperature = (((TEMP_AVG - 1) * temperature) + max31855.readCelsius()) / TEMP_AVG; // accumulate, averaging ratio (for smoother readings)
        ambient = max31855.readInternal(false);
        // AJR TODO 2023-09-12: Disabled for old PID controller code
        // if (l298nhb.active())
        //  l298nhb.setAmbient(ambient);
#endif
      }
    }

    if (!is_running)
    {
      is_running = true;
      check_dtr = Serial.dtr(); // if high, detect low to stop
      ad9851.wakeup();
#if USE_MCP9808
      mcp9808.wakeup(); // wake with no delay
#endif
      ledWrite(LED_RED_PIN, HIGH);
      ledWrite(LED_BLUE_PIN, HIGH);
      ledWrite(LED_ORANGE_PIN, HIGH);
    }

    if (identifying)
    {
      ledWrite(LED_WHITE_PIN, LOW);
      identifying = false;
      tft_identify(identifying); // if 'stream' called alone, 'measure' will be overwritten
      if (streaming)
        tft_measure(); // just for good measure, not a valid use-case scenario in software!
    }

    pre_time = micros();

    // check for calibration sweep (if sweep > 10 MHz)
    if (this_freq_stop - this_freq_start > 10000000)
    {
      tft_initialize();

      // divert thread to cal function and do not do dynamic scanning
      calibrate(this_freq_start, this_freq_stop, this_freq_base); // "base" is really step_size here

      tft_idle();
      return;
    }

    // do frequency hopping (if configured)
    if (base_overtones_per_cycle != 0)
    {
      if (base_increment)
      {
        base_overtone_counter++;
        if (base_overtone_counter > base_overtones_per_cycle)
        {
          overtone++;
          overtone++; // AJR 2022-08-25: disable 5th mode when hopping
          if (overtone >= 3)
          {
            overtone = 0;
            base_overtone_counter = 0; // repeat
          }
        }
        else
        {
          overtone = 0;
        }
        base_increment = false;
      }
      if (overtone == 1)
      {
        this_freq_start = freq_start_up;
        this_freq_stop = freq_stop_up;
        this_freq_base = freq_base_up;
      }
      if (overtone == 2)
      {
        this_freq_start = freq_start_down;
        this_freq_stop = freq_stop_down;
        this_freq_base = freq_base_down;
      }
    }
    else
    {
      overtone = 0xFF;
    }

    // Calculate averaging parameters for hopping (if not yet known)
    if (smooth_factor_in_hop == -1)
    {
      if (base_overtones_per_cycle != 0)
        avg_in_hop = avg_in; // max(1, avg_in * 2);
      smooth_factor_in_hop = toSmoothFactor(avg_in_hop);
    }

    /*** START: FIND PEAK ***/
    if (n == 0 || ((overtone == 1 || overtone == 2) && n < base_overtones_per_cycle + 2))
    {
      if (n == 0)
        stream_start = micros();

      byte find_step_size = min(25, step_size); // max out at 25 Hz
      long num_samples = (this_freq_stop - this_freq_start) / find_step_size;
      if (num_samples > 250)
        find_step_size = ceil(num_samples / 250);

      while (max_freq == 0 || max_freq == this_freq_start || max_freq == this_freq_stop)
      {
        max_mag = 0;
        avg_mag = 0;

        unsigned long delta = abs(app_freq - this_freq_stop);
        app_freq = this_freq_stop;
        ad9851.setFreq(app_freq);
        delayMicroseconds(delta / 10000); // waitForAdcToSettle();

        while (true)
        {
          ad9851.setFreq(app_freq);
          delayMicroseconds(5 * find_step_size); // waitForAdcToSettle();

          int app_mag = ad8302.read();

          if (avg_mag == 0)
            avg_mag = app_mag;
          if (overtone == 1 || overtone == 2)
            avg_mag = (((smooth_factor_in_hop - 1) * avg_mag) + app_mag) / smooth_factor_in_hop;
          else
            avg_mag = (((smooth_factor_in - 1) * avg_mag) + app_mag) / smooth_factor_in;
          app_mag = floor(avg_mag);

          if (max_mag <= app_mag /*|| k < avg_in*/)
          {
            num_max_samples = (max_mag == app_mag ? num_max_samples + 1 : 0);
            max_freq = (max_mag == app_mag ? max_freq : app_freq);
            max_mag = app_mag;
            peak_time = micros();
            samples_since_max = 0;
          }
          else
          {
            if (app_mag <= floor(last_mag))
            {
              samples_since_max++;
            }
            else
            {
              samples_since_max = 0;
            }
          }

          if (app_freq <= this_freq_start)
            break;

          app_freq -= find_step_size;
          k++;
          // last_freq = app_freq;
          last_mag = avg_mag;

          // if (Serial.available()) break;
        }

        // allow for break of infinite loop if/when no peaks are found
        if (Serial.available())
          break;
      }

      // DIR flipped when limit reached, so inverse logic first
      dir = -1;

      peak_freq_avg = max_freq + (dir * find_step_size * (num_max_samples / 2));
      peak_mag_avg = max_mag;
      peak_time_avg = peak_time;
      this_freq_start = peak_freq_avg - max_drift_hz_l;
      this_freq_stop = peak_freq_avg + max_drift_hz_r;

      /*** END: FIND PEAK ***/

      if (overtone == 0 || overtone == 0xFF)
      {
        dir = 1;
        peak_freq_left = peak_freq_right = peak_freq_avg * 100;
        peak_freq_l = peak_freq_h = peak_freq_avg;
        peak_mag_l = peak_mag_h = peak_mag_avg;
        peak_time_l = peak_time_h = peak_time_avg;
        freq_start = this_freq_start;
        freq_stop = this_freq_stop;
      }

      if (overtone == 1)
      {
        dir_up = 1;
        peak_freq_left_up = peak_freq_right_up = peak_freq_avg * 100;
        peak_freq_l_up = peak_freq_h_up = peak_freq_avg;
        peak_mag_l_up = peak_mag_h_up = peak_mag_avg;
        peak_time_l_up = peak_time_h_up = peak_time_avg;
        freq_start_up = this_freq_start;
        freq_stop_up = this_freq_stop;
      }

      if (overtone == 2)
      {
        dir_down = 1;
        peak_freq_left_down = peak_freq_right_down = peak_freq_avg * 100;
        peak_freq_l_down = peak_freq_h_down = peak_freq_avg;
        peak_mag_l_down = peak_mag_h_down = peak_mag_avg;
        peak_time_l_down = peak_time_h_down = peak_time_avg;
        freq_start_down = this_freq_start;
        freq_stop_down = this_freq_stop;
      }

      peak_freq_avg *= 100; // peak_freq_[left/right][_[up/down]] and peak_freq_avg are all x100 (think 'Konami Code')
    }

    /*** START: PEAK TO STOP ***/
    last_mag = 0;
    max_mag = 0;
    max_mag_flt = 0;
    avg_mag = 0;
    peak = false;
    k = 0;
    samples_since_max = 0;
    num_max_samples = 0;
    max_freq = 0;
    bool waitNeeded = false;

    if ((overtone == 0 && base_overtone_counter == 0) || n == 0)
    {
      dir = 1;
      app_freq = (peak_freq_left / 100) - 50;
      waitNeeded = true;
    }
    if (overtone == 1)
    {
      dir = 1; // always
      app_freq = (dir == 1) ? (peak_freq_left_up / 100) - 500 : (peak_freq_right_up / 100) + 500;
      waitNeeded = true;
    }
    if (overtone == 2)
    {
      dir = -1; // always
      app_freq = (dir == 1) ? (peak_freq_left_down / 100) - 500 : (peak_freq_right_down / 100) + 500;
      waitNeeded = true;
    }

    if (waitNeeded)
    {
      ad9851.setFreq(app_freq);
      delayMicroseconds(500);
      // waitForAdcToSettle();
    }

    // long freq_test_start = app_freq;

    while (true)
    {
      ad9851.setFreq(app_freq);
      delayMicroseconds(5);
      int app_mag = ad8302.read();

      if (last_mag_2 == 0)
        last_mag_2 = app_mag;
      if (last_mag_1 == 0)
        last_mag_1 = app_mag;
      if (last_mag == 0)
        last_mag = app_mag;
      if (avg_mag == 0)
        avg_mag = app_mag;
      if (overtone == 1 || overtone == 2)
        avg_mag = (((smooth_factor_in_hop - 1) * avg_mag) + app_mag) / smooth_factor_in_hop;
      else
        avg_mag = (((smooth_factor_in - 1) * avg_mag) + app_mag) / smooth_factor_in;
      app_mag = floor(avg_mag);

      if (max_mag_flt <= avg_mag)
      {
        max_mag_flt = avg_mag;
        max_freq = app_freq;
        peak_time = micros();
      }

      if (max_mag <= app_mag /*|| k < avg_in*/)
      {
        num_max_samples = (max_mag == app_mag ? num_max_samples + 1 : 0);
        // max_freq = (max_mag == app_mag && max_freq ? max_freq : app_freq);
        max_mag = app_mag;
        // peak_time = micros();
        samples_since_max = 0;
      }
      else
      {
        if (app_mag <= floor(last_mag))
        {
          samples_since_max++;
        }
        else
        {
          samples_since_max = 0;
        }
      }

      // if (samples_since_max > 3) break;
      if (dir == 1)
      {
        if (max_freq != max_mag)
        {
          if (overtone == 1 && app_freq > peak_freq_h_up)
            peak = true;
          else if (overtone == 2 && app_freq > peak_freq_h_down)
            peak = true;
          else if (app_freq > peak_freq_h)
            peak = true;
        }
      }
      else
      {
        if (max_freq != max_mag)
        {
          if (overtone == 1 && app_freq < peak_freq_l_up)
            peak = true;
          else if (overtone == 2 && app_freq < peak_freq_l_down)
            peak = true;
          else if (app_freq < peak_freq_l)
            peak = true;
        }
      }
      // float avg_mag_dB = ((avg_mag - this_freq_base) * (3.3 / 8191) - 0.9) / 0.029 + 31;
      // float max_mag_dB = ((max_mag - this_freq_base) * (3.3 / 8191) - 0.9) / 0.029 + 31;
      // if (peak && avg_mag_dB <= max_mag_dB - 1.0)
      if (peak && floor(last_mag - this_freq_base) <= floor(power_point * (max_mag - this_freq_base)))
      {
        // client->print("avg mag = ");
        // client->println(avg_mag_dB);
        // client->print("max mag = ");
        // client->println(max_mag_dB);

        db_actual = floor(100 * (avg_mag - this_freq_base));
        db_desired = floor(100 * power_point * (max_mag - this_freq_base)) - db_actual;
        db_actual = floor(100 * (last_mag_2 - this_freq_base)) - db_actual;
        /*client->print(db_desired);
          client->print(" / ");
          client->println(db_actual);
          client->print(app_freq * 100);
          client->print(" / ");
          client->print(dir * step_size * (100 * db_desired / db_actual));
          client->print(" / ");*/
        unsigned long temp = app_freq;
        temp *= 100;
        temp -= (3 * dir * step_size * (100 * db_desired / db_actual));
        db_actual = temp;
        // client->println(db_actual);
        break;
      }
      if (k > 0 && app_freq <= this_freq_start)
      {
        max_freq = this_freq_start;
        break;
      }
      if (k > 0 && app_freq >= this_freq_stop)
      {
        max_freq = this_freq_stop;
        break;
      }

      app_freq += dir * step_size;
      k++;
      // last_freq = app_freq;

      // shift last 3x magnitude buffer
      last_mag_2 = last_mag_1;
      last_mag_1 = last_mag;
      last_mag = avg_mag;

      bool updateData = ((micros() - last_time) > swSpeed_us && new_sample);
      if (updateData)
      {
        last_time = micros();
        base_increment = true;
        new_sample = false;

        client->printf("%s;%u;%u;%u;%u;%i;%.2f;%.2f%s",
                       "QI",                                                   // format version
                       n++,                                                    // sequence
                       (peak_time_avg - stream_start) / 100,                   // timestamp
                       peak_mag_avg,                                           // peak mag
                       peak_freq_avg / 100,                                    // peak freq
                       l298nhb_status * l298nhb.getPower(),                    // status/power (temp branch only)
                       ambient,                                                // ambient temp (temp branch only)
                       round(temperature / TEMP_RESOLUTION) * TEMP_RESOLUTION, // temperature (https://arduino.stackexchange.com/a/28469)
                       "\n");                                                  // end of sample
      }

      // if (Serial.available()) break;
    }

    // long freq_test_stop = app_freq;
    // client->printf("start = %u, stop = %u, avg = %u\n", freq_test_start, freq_test_stop, (freq_test_start + freq_test_stop) / 2);

    long new_freq = max_freq + (dir * step_size * (num_max_samples / 2));

    // Calculate averaging parameters for hopping (if not yet known)
    if (smooth_factor_out_hop == -1)
    {
      if (base_overtones_per_cycle != 0)
        avg_out_hop = max(1, avg_out + 1); // max(1, ceil(avg_out / base_overtones_per_cycle));
      smooth_factor_out_hop = toSmoothFactor(avg_out_hop);
    }

    check_and_correct_micros_rollover(); // adjusts time variables on rollover event
    if (dir == 1)
    {
      if (overtone == 1)
      {
        peak_freq_right_up = ((((2 * smooth_factor_out_hop) - 1) * peak_freq_right_up) + db_actual) / (2 * smooth_factor_out_hop);
        peak_freq_h_up = (((smooth_factor_out_hop - 1) * peak_freq_h_up) + new_freq) / smooth_factor_out_hop;
        peak_mag_l_up = (((smooth_factor_out_hop - 1) * peak_mag_l_up) + max_mag) / smooth_factor_out_hop;
        peak_time_l_up = (((smooth_factor_out_hop - 1) * peak_time_l_up) + peak_time) / smooth_factor_out_hop;
        dir_up = -1; // switch dir
      }
      else if (overtone == 2)
      {
        peak_freq_right_down = ((((2 * smooth_factor_out_hop) - 1) * peak_freq_right_down) + db_actual) / (2 * smooth_factor_out_hop);
        peak_freq_h_down = (((smooth_factor_out_hop - 1) * peak_freq_h_down) + new_freq) / smooth_factor_out_hop;
        peak_mag_l_down = (((smooth_factor_out_hop - 1) * peak_mag_l_down) + max_mag) / smooth_factor_out_hop;
        peak_time_l_down = (((smooth_factor_out_hop - 1) * peak_time_l_down) + peak_time) / smooth_factor_out_hop;
        dir_down = -1; // switch dir
      }
      else // 0x00 or 0xFF
      {
        peak_freq_right = ((((freq_factor * smooth_factor_out) - 1) * peak_freq_right) + db_actual) / (freq_factor * smooth_factor_out);
        peak_freq_h = (((smooth_factor_out - 1) * peak_freq_h) + new_freq) / smooth_factor_out;
        peak_mag_l = (((smooth_factor_out - 1) * peak_mag_l) + max_mag) / smooth_factor_out;
        peak_time_l = (((smooth_factor_out - 1) * peak_time_l) + peak_time) / smooth_factor_out;
        dir = -1; // switch dir
      }
      // client->printf("low = %u\n", new_freq);
    }
    else
    {
      if (overtone == 1)
      {
        peak_freq_left_up = ((((2 * smooth_factor_out_hop) - 1) * peak_freq_left_up) + db_actual) / (2 * smooth_factor_out_hop);
        peak_freq_l_up = (((smooth_factor_out_hop - 1) * peak_freq_l_up) + new_freq) / smooth_factor_out_hop;
        peak_mag_h_up = (((smooth_factor_out_hop - 1) * peak_mag_h_up) + max_mag) / smooth_factor_out_hop;
        peak_time_h_up = (((smooth_factor_out_hop - 1) * peak_time_h_up) + peak_time) / smooth_factor_out_hop;
        dir_up = 1; // switch dir
      }
      else if (overtone == 2)
      {
        peak_freq_left_down = ((((2 * smooth_factor_out_hop) - 1) * peak_freq_left_down) + db_actual) / (2 * smooth_factor_out_hop);
        peak_freq_l_down = (((smooth_factor_out_hop - 1) * peak_freq_l_down) + new_freq) / smooth_factor_out_hop;
        peak_mag_h_down = (((smooth_factor_out_hop - 1) * peak_mag_h_down) + max_mag) / smooth_factor_out_hop;
        peak_time_h_down = (((smooth_factor_out_hop - 1) * peak_time_h_down) + peak_time) / smooth_factor_out_hop;
        dir_down = 1; // switch dir
      }
      else // 0x00 or 0xFF
      {
        peak_freq_left = ((((freq_factor * smooth_factor_out) - 1) * peak_freq_left) + db_actual) / (freq_factor * smooth_factor_out);
        peak_freq_l = (((smooth_factor_out - 1) * peak_freq_l) + new_freq) / smooth_factor_out;
        peak_mag_h = (((smooth_factor_out - 1) * peak_mag_h) + max_mag) / smooth_factor_out;
        peak_time_h = (((smooth_factor_out - 1) * peak_time_h) + peak_time) / smooth_factor_out;
        dir = 1; // switch dir
      }
      // client->printf("high = %u\n", new_freq);
    }
    /*** END: PEAK TO STOP ***/

    if (overtone == 1) // always DIR = 1
    {
      peak_time_avg = peak_time_l_up;     //(peak_time_h_up / 2) + (peak_time_l_up / 2);
      peak_freq_avg = peak_freq_right_up; //(peak_freq_left_up / 2) + (peak_freq_right_up / 2);
      peak_mag_avg = peak_mag_l_up;       //(peak_mag_h_up / 2) + (peak_mag_l_up / 2);
      freq_overshoot = peak_freq_h_up;    //(peak_freq_h_up / 2) - (peak_freq_l_up / 2);
      left = peak_freq_left_up;
      right = peak_freq_right_up;
    }
    else if (overtone == 2) // always DIR = -1
    {
      peak_time_avg = peak_time_h_down;    //(peak_time_h_down / 2) + (peak_time_l_down / 2);
      peak_freq_avg = peak_freq_left_down; //(peak_freq_left_down / 2) + (peak_freq_right_down / 2);
      peak_mag_avg = peak_mag_h_down;      //(peak_mag_h_down / 2) + (peak_mag_l_down / 2);
      freq_overshoot = peak_freq_l_down;   //(peak_freq_h_down / 2) - (peak_freq_l_down / 2);
      left = peak_freq_left_down;
      right = peak_freq_right_down;
    }
    else // 0x00 or 0xFF
    {
      // NOTE: this can cause time jump if one variable has rolled but the other has not
      // To resolve, 'check_and_correct_micros_rollover()' will detect this condition
      peak_time_avg = (peak_time_h / 2) + (peak_time_l / 2);
      peak_freq_avg = (peak_freq_left / 2) + (peak_freq_right / 2);
      peak_mag_avg = (peak_mag_h / 2) + (peak_mag_l / 2);
      freq_overshoot = (peak_freq_h / 2) - (peak_freq_l / 2);
      left = peak_freq_left;
      right = peak_freq_right;
    }

    freq_overshoot = 0;
    new_sample = true;

    bool updateTemp = ((millis() - last_temp) > 2000);
    if (updateTemp)
    {
      last_temp = millis();

      // measure temperature
#if USE_MCP9808
      temperature = mcp9808.readTempC();
#endif
#if USE_MAX31855
      if (isnan(temperature))
        temperature = max31855.readCelsius();
      else
        temperature = (((TEMP_AVG - 1) * temperature) + max31855.readCelsius()) / TEMP_AVG; // accumulate, averaging ratio (for smoother readings)
      ambient = max31855.readInternal(false);
      // AJR TODO 2023-09-12: Disabled for old PID controller code
      // if (l298nhb.active())
      //  l298nhb.setAmbient(ambient);
#endif
    }

    bool updateData = ((micros() - last_time) > swSpeed_us && !streaming);
    if (updateData)
    {
      last_time = micros();
      base_increment = true;

      client->printf("%s;%u;%u;%u;%u;%i;%.2f;%.2f%s",
                     "QI",                                                   // format version
                     n++,                                                    // sequence
                     (peak_time_avg - stream_start) / 100,                   // timestamp
                     peak_mag_avg,                                           // peak mag
                     peak_freq_avg / 100,                                    // peak freq
                     l298nhb_status * l298nhb.getPower(),                    // status/power (temp branch only)
                     ambient,                                                // ambient temp (temp branch only)
                     round(temperature / TEMP_RESOLUTION) * TEMP_RESOLUTION, // temperature (https://arduino.stackexchange.com/a/28469)
                     "\n");                                                  // end of sample
    }

    if (DEBUG)
    {
      Serial.flush();
      Serial.println();
      Serial.print("Total samples:   ");
      Serial.println(((this_freq_stop - this_freq_start) / step_size) + 1);
      Serial.print("Sweep time (ms): ");
      Serial.print((last_time - pre_time) / 1000);
      Serial.print(".");
      Serial.println((last_time - pre_time) % 1000);
    }

    // report stream timeout event
    if (quack_counter > quack_interval || (peak_time_avg - stream_start) / 100 > 36000000) // 1 million samples or 1 hour runtime, whichever comes first
    {
      client->println("QUACK!");
      quack_counter = quack_interval + 1; // force timeout
    }

    if ((l298nhb_auto_off_at - millis()) < L298NHB_AUTOOFF && l298nhb.active()) // not yet time to auto off
    {
      l298nhb_auto_off_at = millis() + L298NHB_AUTOOFF; // calculate time to auto off
      // special case in event of rollover in prior line math
      if (l298nhb_auto_off_at == 0) 
      {
        l298nhb_auto_off_at = 1;
      }
    }

    // end sweep
  }
  else if (is_running)
  {
    // power things down
    is_running = false;
    ad9851.shutdown();
#if USE_MCP9808
    mcp9808.shutdown();
#endif
    if (hw_error)
      ledWrite(LED_ORANGE_PIN, HIGH);
    else
      ledWrite(LED_ORANGE_PIN, LOW);
    ledWrite(LED_RED_PIN, LOW);
    ledWrite(LED_BLUE_PIN, LOW);
    ledWrite(LED_WHITE_PIN, LOW);
    ledWrite(LED_SEGMENT_DP, LOW);
  }
  else
  {
    // idle mode - blink LED (speed dependent on port status)
    // Assume USB port closed - slower, quicker blink rate
    unsigned short blink_rate = 3000;
    unsigned short blink_on = 100;

    if (Serial.dtr() || identifying)
    {
      if (identifying && (stop_identify_at - millis()) > 60000)
      {
        ledWrite(LED_WHITE_PIN, LOW);
        identifying = false;
        tft_identify(identifying);
      }

      // USB port open - faster, longer blink rate
      blink_rate = 500;
      blink_on = 250;
    }

    bool led_state = (getSystemTime() % blink_rate <= blink_on) ? HIGH : LOW;
    //    ledWrite(LED_RED_PIN, led_state);
    //    ledWrite(LED_BLUE_PIN, led_state);
    if (!l298nhb.active())
      ledWrite(LED_SEGMENT_DP, led_state);
    if (l298nhb_auto_off_at == 0)
    {
      ledWrite(LED_BLUE_PIN, led_state);
      ledWrite(LED_ORANGE_PIN, LOW);
    }
    else
      ledWrite(LED_ORANGE_PIN, led_state);
    if (Serial.dtr() || identifying)
    {
      ledWrite(LED_WHITE_PIN, led_state);
      //      if (identifying) // disable due to photosensitive effects
      //      {
      //        tft_wakeup();
      //        tft.invertDisplay(led_state);
      //      }
    }
    else
      ledWrite(LED_WHITE_PIN, LOW);

    bool show_errors = true;
    unsigned long now_us = micros();
    bool show_screensaver = ((now_us - time_of_last_msg) > sleep_timer_delay); // screensaver after 15 mins of idle time
#if USE_L298NHB
    show_errors = !l298nhb.active();
#endif

    if (show_errors && show_screensaver && !sleep_running)
    {
      //      Serial.println("Sleep timer auto started"); // debug
      sleepTimer.begin([]
                       { tft_screensaver(); },
                       sleepTimerInt_us);
      tft_screensaver(); // show immediately, don't wait for interval
      sleep_running = true;
    }
    else if (now_us < sleep_timer_delay &&             // When a rollover has occurred - AND
             (time_of_last_msg == 0 ||                 // (a message was never received - OR -
              time_of_last_msg >= sleepTimerInt_us) && // it was not received recently) - AND
             sleep_running)                            // the screensaver is running:
    {
      // Then handle the micros() rollover event (every ~45 mins) to prevent sleep ending
      // NOTE: If the message WAS received recently (during the first 'sleepTimerInt_us'
      // after a rollover event, then condition #3 above will allow sleep to end now
      //      Serial.println("We've had a micros() rollover event!"); // debug
      time_of_last_msg = 0; // it's a new world, act like a message was never received
      sleep_timer_delay = sleepTimerInt_us;
      delay(1000);
    }
    else if (!show_screensaver && sleep_running)
    {
      //      Serial.println("Screensaver ended!"); // debug
      sleepTimer.end();
      sleep_running = false;
      sleep_timer_delay = sleepTimerDelay_us;

      if (time_of_last_msg < sleepTimerInt_us && !sleep_running)
      {
        // this will prevent our world view from becoming misaligned with reality
        // (i.e. we could end sleep prematurely due to rollover, not a message rx)
        time_of_last_msg = sleepTimerInt_us;
      }

      if (!identifying)
        tft_idle();
    }

    //    if (show_errors)
    //    {
    //      if (hw_error) segmentDisplay.displayCharacter('E');
    //      if (false && net_error) digitalWrite(LED_SEGS_END, LED_SEGS_ANODE);
    //      else digitalWrite(LED_SEGS_END, led_state);
    //    }
  }

#if USE_L298NHB
  uint16_t tec_rate = 500;
  /* L298NHB Task Update */
  if (l298nhb.active())
  {
    digitalWrite(TEMP_CIRCUIT, HIGH); // turn on temp circuit
    digitalWrite(TEC_SENSE, HIGH);

    bool updateTemp = ((millis() - last_temp) > tec_rate);
    if (updateTemp)
    {
#ifdef SHUTDOWN_TEC_FOR_TEMP_READS
      byte restore = l298nhb.getPower();
      l298nhb.setPower(0); // off
                           // digitalWrite(TEMP_CIRCUIT, LOW); // turn off fan
                           // digitalWrite(FAN_HIGH_LOW, LOW);
                           // delay(100); // wait for noise to settle
#endif
      last_temp = millis();
      // measure temperature
#if USE_MCP9808
      mcp9808.wakeup();
      temperature = mcp9808.readTempC();
      mcp9808.shutdown();
#endif
#if USE_MAX31855
      if (isnan(temperature))
        temperature = max31855.readCelsius();
      else
        temperature = (((TEMP_AVG - 1) * temperature) + max31855.readCelsius()) / TEMP_AVG; // accumulate, averaging ratio (for smoother readings)
      ambient = max31855.readInternal(false);
      // AJR TODO 2023-09-12: Disabled for old PID controller code
      // l298nhb.setAmbient(ambient);
#endif
#ifdef SHUTDOWN_TEC_FOR_TEMP_READS
      // digitalWrite(TEMP_CIRCUIT, HIGH); // turn on fan
      // digitalWrite(FAN_HIGH_LOW, HIGH);
      l298nhb.setPower(restore); // resume
#endif
    }
    bool updateTEC = ((millis() - l298nhb_task_timer) > tec_rate);
    if (isnan(temperature))
    { // read failed, MAX31855 chip error
      hw_error = true;
      updateTEC = false;
      l298nhb.shutdown();
      // digitalWrite(FAN_HIGH_LOW, LOW);
      //      tft_tempcontrol(); // draw "OFF" state
      tft_cooldown_start(); // change status to "cooldown" and start countdown
    }
    // SKIP THIS: TFT_TEMPCONTROL() NOW USES HW_ERROR TO FLAG LCD ERROR MESSAGE WRITE/CLEAR
    //    else if (hw_error)
    //    { // clear error on successful TEMP read
    //      hw_error = false;
    //    }
    if (updateTEC)
    {
      l298nhb_status = l298nhb.update(temperature);
      l298nhb_task_timer = millis();
#if USE_MAX31855
      max31855.setMode(l298nhb_status);
#endif

      if (!streaming)
      {
        //        char H_or_C = (l298nhb_status == -1) ? 'C' : 'X'; // 'X' looks like 'H' on display
        ledWrite((l298nhb_status == -1) ? LED_BLUE_PIN : LED_RED_PIN, l298nhb.getPower());
        ledWrite((l298nhb_status == -1) ? LED_RED_PIN : LED_BLUE_PIN, LOW); // other mode off
        // blink DP while temp cycling, steady on when stable
        //        bool next_state = digitalRead(LED_SEGS_END);
        //        if (!LED_SEGS_ANODE) next_state = !next_state;
        if (l298nhb.targetReached())
        {
          // requires pinMode be digital (TODO to ensure this mode first using ledWrite())
          digitalToggle(LED_WHITE_PIN); // blink it
          //          digitalWrite(LED_WHITE_PIN, !digitalRead(LED_WHITE_PIN)); // blink it
          //          next_state = true;
        }

        //        segmentDisplay.displayCharacter(H_or_C); // indicate TEC mode on display
        //        segmentDisplay.displayDecimalPoint(next_state);
        digitalToggle(LED_SEGMENT_DP);
        tft_tempcontrol();
      }
      // else
      // {
      //   tft_tempbusy();
      // }

      // // if running at full-power cold or still cycling to target
      // if (l298nhb.getPower() == 255 || !l298nhb.targetReached())
      // {
      //   // digitalWrite(FAN_HIGH_LOW, HIGH); // high speed fan
      // }
      // else // if (!l298nhb.getSignal()) // if running in heat mode
      // {
      //   // digitalWrite(FAN_HIGH_LOW, LOW); // low speed fan
      // }
    }
    if ((l298nhb_auto_off_at - millis()) > L298NHB_AUTOOFF && l298nhb.active()) // time to auto off
    {
      l298nhb.shutdown();
      // digitalWrite(FAN_HIGH_LOW, LOW); // low speed fan
      //      tft_tempcontrol(); // draw "OFF" state
      tft_cooldown_start(); // change status to "cooldown" and start countdown
    }
  }
  else if (l298nhb_auto_off_at != 0) // in cool-down mode
  {
    //    segmentDisplay.displayCharacter(' '); // indicate "off" while in this mode
    bool updateTEC = ((millis() - l298nhb_task_timer) > tec_rate * 2);
    if (updateTEC)
    {
      l298nhb_task_timer = millis();
      // blink DP while temp cycling, steady on when stable
      //      bool next_state = digitalRead(LED_SEGS_END);
      //      if (!LED_SEGS_ANODE) next_state = !next_state;
      //      segmentDisplay.displayDecimalPoint(next_state); // toggle on/off
      //      Serial.println(digitalRead(LED_BLUE_PIN) ? "Red LED" : "Blue LED"); // test only
      ledWrite(LED_RED_PIN, digitalRead(LED_BLUE_PIN) ? HIGH : LOW);
      ledWrite(LED_BLUE_PIN, digitalRead(LED_BLUE_PIN) ? LOW : HIGH);
      ledWrite(LED_WHITE_PIN, LOW);

      if (!streaming)
        tft_cooldown(); // update countdown, already started

      if ((l298nhb_auto_off_at - millis()) > L298NHB_COOLDOWN) // time to end cooldown
      {
        l298nhb_auto_off_at = 0;
        //        segmentDisplay.displayDecimalPoint(false);
        digitalWrite(TEMP_CIRCUIT, LOW); // turn off fan
        digitalWrite(TEC_SENSE, LOW);
        ledWrite(LED_RED_PIN, LOW);
        ledWrite(LED_BLUE_PIN, LOW);
        tft_idle();
      }
    }
  }
#endif
}

/************************** FUNCTION ***************************/

void calibrate(long start, long stop, long step)
{
  // delta enoding helpers
  int lastVal_mag = 0;
  bool everyOther = true;
  bool firstHit = true;
#if defined(USB_RAWHID)
  char buffer[3];
#endif

  // start sweep cycle measurement
  for (int count = start; count <= stop + step; count += step)
  {
    // do the magic ! waiting for the ADC measure
    // also doing serial TX here allows signal to settle
    //      some after SetFreq() and before ADC measure!
    if (count == start)
    {

      // test code
      //      if (false) {
      //        int freq = 25000000;
      //        ad9851.setFreq(freq); // set test frequency
      //        while (1) {} // wait forever
      //      }

      // set AD9851 DDS current frequency
      ad9851.setFreq(count); // updates: set_freq, wait_delay, last_jump
      delayMicroseconds(500);

      if (ENCODE_SERIAL)
      {
        bool reportTemp = true; //((millis() - last_temp) > 2000);

        // "Q" denotes system: QATCH
        // "A" denotes format: mag, phase, temp
        // "B" denotes format: mag, phase
        // "C" denotes format: mag, temp
        // "D" denotes format: mag
        // "E" and "F" are not supported
        // "G" denotes format: mag deltas, raw temp
        // "H" denotes format: mag deltas only
        client->print("Q");
        if (!USE_DELTAS)
        {
          client->print(reportTemp ? "C" : "D");
        }
        else // deltas
        {
          client->print(reportTemp ? "G" : "H");
#if defined(USB_RAWHID)
          // sprintf(buffer, "%02x", overtone);
          // client->print(overtone);
          client->print(" "); // byte as ASCII format specifier: ' ' (space)
#else
          client->write(overtone); // freq-hop
#endif
        }
      }
    }

    // measure gain phase
    int app_mag = 0;

    // ADC measure and averaging
    for (int i = 0; i < AVERAGE_SAMPLE; i++)
    {
      app_mag += ad8302.read();
    }

    // set AD9851 DDS current frequency
    ad9851.setFreq(count + step); // updates: set_freq, wait_delay, last_jump

    // averaging (cast to double)
    double measure_mag = 1.0 * app_mag / AVERAGE_SAMPLE;

    if (count >= start + step) // not the first freq in the sweep
    {
      // serial write data (all values)
      if (USE_DELTAS)
      {
        if (firstHit)
        {
          short mag_int = (short)(measure_mag);
          byte mag_int0 = (mag_int & 0x00FF) >> 0;
          byte mag_int1 = (mag_int & 0xFF00) >> 8;
#if defined(USB_RAWHID)
          sprintf(buffer, "%02x", mag_int1);
          client->print(buffer); // byte as ASCII
          sprintf(buffer, "%02x", mag_int0);
          client->print(buffer); // byte as ASCII
#else
          client->write(mag_int1);
          client->write(mag_int0);
#endif
          targetDelta_mag = 0;
          lastLastVal_mag = (int)mag_int;
          firstHit = false;
        }
        else if (everyOther)
        {
          lastVal_mag = (int)measure_mag;
          everyOther = false;
        }
        else
        {
          byte delta_mag = deltaMag(lastVal_mag, (int)measure_mag);
#if defined(USB_RAWHID)
          sprintf(buffer, "%02x", delta_mag);
          client->print(buffer); // byte as ASCII
#else
          client->write(delta_mag);
#endif
          everyOther = true;
        }
      }
      else if (ENCODE_SERIAL)
      {
        short mag_int = (short)(measure_mag);
        byte mag_int0 = (mag_int & 0x00FF) >> 0;
        byte mag_int1 = (mag_int & 0xFF00) >> 8;
#if defined(USB_RAWHID)
        sprintf(buffer, "%02x", mag_int1);
        client->print(buffer); // byte as ASCII
        sprintf(buffer, "%02x", mag_int0);
        client->print(buffer); // byte as ASCII
#else
        client->write(mag_int1);
        client->write(mag_int0);
#endif
      }
      else
      {
        client->print(measure_mag);
        client->print(";");
      }
      delayMicroseconds(50);
    }
  }

  // measure temperature
#if USE_MCP9808
  temperature = mcp9808.readTempC();
#endif
#if USE_MAX31855
  if (isnan(temperature))
    temperature = max31855.readCelsius();
  if (isnan(temperature)) // if still NAN, use internal (ambient)
    temperature = max31855.readInternal(false);
  else
    temperature = max31855.readCelsius(); // (((TEMP_AVG - 1)*temperature) + max31855.readCelsius()) / TEMP_AVG;
#endif

  // serial write temperature data at the end of the sweep
  if (ENCODE_SERIAL)
  {
    short temp_int = (short)(temperature * 100);
    byte temp_int0 = (temp_int & 0x00FF) >> 0;
    byte temp_int1 = (temp_int & 0xFF00) >> 8;
#if defined(USB_RAWHID)
    sprintf(buffer, "%02x", temp_int1);
    client->print(buffer); // byte as ASCII
    sprintf(buffer, "%02x", temp_int0);
    client->print(buffer); // byte as ASCII
#else
    client->write(temp_int1); // freq-hop
    client->write(temp_int0); // freq-hop
#endif
  }
  else
  {
    client->print(temperature);
    client->print(";");
  }

  // print termination char EOM
  if (!ENCODE_SERIAL)
  {
    client->println("s");
  }

  client->flush(); // Wait for the transmission of outgoing serial data to complete.
}

/* Send deltas for magnitude */
byte deltaMag(int val1, int val2)
{
  int compress = 4; // compression ratio (MUST MATCH SW)
  byte out = 0;
  int sign = 0;

  // calculate val1
  targetDelta_mag += (val1 - lastLastVal_mag);
  if (targetDelta_mag > 7 * compress)
  {
    // too big to encode now, indicate max and send rest later
    out |= 0x70; // 0111 0000 (7)
    targetDelta_mag -= (7 * compress);
  }
  else if (targetDelta_mag < -8 * compress)
  {
    // too small to encode now, indicate min and send rest later
    out |= 0x80; // 1000 0000 (-8)
    targetDelta_mag -= (-8 * compress);
  }
  else
  {
    // entire delta can be encoded now, so do that
    out |= ((targetDelta_mag / compress) << 4);
    sign = (targetDelta_mag >= 0 ? 1 : -1);
    targetDelta_mag = (targetDelta_mag % (sign * compress));
  }

  // calculate val2
  targetDelta_mag += (val2 - val1);
  if (targetDelta_mag > 7 * compress)
  {
    // too big to encode now, indicate max and send rest later
    out |= 0x07; // 0000 0111 (7)
    targetDelta_mag -= (7 * compress);
  }
  else if (targetDelta_mag < -8 * compress)
  {
    // too small to encode now, indicate min and send rest later
    out |= 0x08; // 0000 1000 (-8)
    targetDelta_mag -= (-8 * compress);
  }
  else
  {
    // entire delta can be encoded now, so do that
    out |= ((targetDelta_mag / compress) & 0x0F);
    sign = (targetDelta_mag >= 0 ? 1 : -1);
    targetDelta_mag = (targetDelta_mag % (sign * compress));
  }

  // store last val for next time
  lastLastVal_mag = val2;

  // return encoded byte
  return out;
}

float toSmoothFactor(byte avg_samples)
{
  // This equation converts the desired sample averaging count
  // to a cooresponding smoothing factor for additive smoothing
  // where the resulting average settles out to within 0.001 of
  // the actual value at 'x' sample specified by 'avg_samples'.
  //
  // i.e. For 10 sample averaging, the smoothing factor is 1.5.
  //
  // NOTE: IT MUST BE THAT 'smoothing factor >= 1.0' so if/when
  // 'avg_samples < 6' the resulting smoothing factor will be 1.

  return max(1, (0.1 * avg_samples) + 0.5);
}

void waitForAdcToSettle(void) // used only by "TEMP CHECK" cmd
{
  float avg_mag_settle = 0;
  int app_mag;
  int compare_to = 0;
  int j = 0;
  unsigned long start_us = micros();

  // timeout after 10 ms
  while (micros() - start_us < 10000)
  {
    app_mag = ad8302.read();

    if (avg_mag_settle == 0)
      avg_mag_settle = app_mag;
    avg_mag_settle = (((smooth_factor_in - 1) * avg_mag_settle) + app_mag) / smooth_factor_in;
    app_mag = floor(avg_mag_settle);

    if (abs(app_mag - compare_to) >= 10)
    {
      compare_to = app_mag;
      j = 0;
    }
    else if (j++ >= avg_in)
      break;
  }
}

bool check_and_correct_micros_rollover(void)
{
  // adjusts time variables on rollover event
  if (peak_time < peak_time_avg)
  {
    //Serial.println("A rollover event has occurred!"); // DEBUG ONLY
    peak_time_l_up = peak_time_l_down = peak_time_l = 0;
    peak_time_h_up = peak_time_h_down = peak_time_h = 0;
    return true;
  }
  return false;
}

void stopStreaming(void)
{
  if (streaming)
    tft_idle();      // only redraw UI if active
  streaming = false; // require "STREAM" cmd again
  message = false;   // one-shot only (not repeatedly)
  new_sample = false;
  last_temp = 0; // require temp on next sweep data
  n = 0;
  max31855.useOffsetM(false);
}

/// @note removed instances of this being called, it's deprecated
// byte EEPROM_read_pid(void)
// {
//   int old_address = 0; // PID
//   int new_address = (EEPROM.length() - 1) - old_address;
//   EEPROM_pid = EEPROM.read(new_address);

//   if (millis() > 1000)
//   {
//     Serial.printf("Set PID = %X\n", EEPROM_pid);
//   }

//   return EEPROM_pid;
// }

/************************** ILI9341 ****************************/

#if USE_ILI9341

void tft_wakeup()
{
  if (HW_REV_MATCH(HW_REVISION_0)) // || PID_IS_SECONDARY(NVMEM.pid))
    return;

  // Serial.println("Writing LCD!"); // debug ONLY

  tft.begin();
  tft.setRotation(TFT_ROTATION);
}

void tft_screensaver()
{
  if (HW_REV_MATCH(HW_REVISION_0) || PID_IS_SECONDARY(NVMEM.pid))
    return;

  // This will auto-enable after 15 minutes of no activity
  uint16_t x, y;

  tft_wakeup();
  tft.fillScreen(ILI9341_BLACK);
  //  tft.setTextColor(0x1111);
  //  tft.setTextSize(2);
  //  tft.setCursor(3, 3);
  //  tft.print(" FW: ");
  //  tft.print(CODE_VERSION);
  //  tft.print(" (");
  //  tft.print(RELEASE_DATE);
  //  tft.print(")");
  //  tft.setTextSize(5);
  //  tft.setCursor(10, 100);
  //  tft.print("SLEEP MODE");
  x = random(TFT_WIDTH - qatch_icon.width);
  y = random(TFT_HEIGHT - qatch_icon.height);
  tft.writeRect(x, y, qatch_icon.width, qatch_icon.height, (uint16_t *)(qatch_icon.pixel_data));
}

void tft_splash(bool dp)
{
  if (HW_REV_MATCH(HW_REVISION_0)) // || PID_IS_SECONDARY(NVMEM.pid))
    return;

  uint16_t x, y, h, w, pad;

  tft_wakeup();
  tft.fillScreen(ILI9341_BLACK);
  //  tft.setTextColor(dp ? ILI9341_BLACK : ILI9341_RED);
  //  tft.setFontAdafruit(); // default console font
  //  tft.setTextSize(2);
  //  tft.writeRect(0, 0, boot_splash.width, boot_splash.height, (uint16_t*)(boot_splash.pixel_data));
  h = 20; // line height
  pad = 5;

  tft_wakeup();
  tft.fillScreen(QATCH_GREY_BG);

  if (dp && !PID_IS_SECONDARY(NVMEM.pid))
  {
    tft_error = (tft.readPixel(tft.width() / 2, tft.height() / 2) != QATCH_GREY_BG);
  }

  x = ICON_X;
  y = ICON_Y; // image size is 100x100
  tft.writeRect(x, y, splash_icon.width, splash_icon.height, (uint16_t *)(splash_icon.pixel_data));

  x = TEXT_X;
  y = TEXT_Y; // image size is 256x48
  tft.writeRect(x, y, nanovisQ_grey.width, nanovisQ_grey.height, (uint16_t *)(nanovisQ_grey.pixel_data));

  // NOTE: these text params are also used by busyTimerTask();
  tft.setTextColor(dp ? ILI9341_BLACK : ILI9341_RED);
  tft.setFontAdafruit(); // default console font
  tft.setTextSize(2);

  String line1 = dp ? CODE_VERSION : "--- UPDATING FIRMWARE ---";
  char buff1[line1.length() + 1]; // trailing NULL
  line1.toCharArray(buff1, sizeof(buff1));
  w = tft.measureTextWidth(buff1);
  tft.setCursor((TFT_WIDTH - w) / 2, TFT_HEIGHT - (2 * h) - pad);
  tft.print(line1);

  String line2 = dp ? RELEASE_DATE : "Do NOT power cycle device!";
  char buff2[line2.length() + 1]; // trailing NULL
  line2.toCharArray(buff2, sizeof(buff2));
  w = tft.measureTextWidth(buff2);
  tft.setCursor((TFT_WIDTH - w) / 2, TFT_HEIGHT - (1 * h) - pad);
  tft.print(line2);

  //  String code_version = CODE_VERSION; // now with null-terminator
  //  w = tft.measureTextWidth(CODE_VERSION, code_version.length());
  //  tft.setCursor((TFT_WIDTH - w) / 2, TFT_HEIGHT - (2 * h) - pad_bottom);
  //  tft.print(CODE_VERSION);
  //
  //  String release_date = RELEASE_DATE; // now with null-terminator
  //  w = tft.measureTextWidth(RELEASE_DATE, release_date.length());
  //  tft.setCursor((TFT_WIDTH - w) / 2, TFT_HEIGHT - (1 * h) - pad_bottom);
  //  tft.print(RELEASE_DATE);
}

void tft_identify(bool identifying)
{
  if (HW_REV_MATCH(HW_REVISION_0) || PID_IS_SECONDARY(NVMEM.pid))
    return;

  tft_wakeup();
  tft_idle();
  if (identifying)
  {
    tft.invertDisplay(true);
    tft.fillRect(0, 0, TFT_WIDTH, 20, ILI9341_BLACK);   // inverted, shows as WHITE
    tft.drawFastHLine(0, 21, TFT_WIDTH, ILI9341_WHITE); // inverted, shows as BLACK
    tft.setTextColor(ILI9341_WHITE);                    // inverted, shows as BLACK
    tft.setFontAdafruit();                              // default console font
    tft.setTextSize(2);
    tft.setCursor(3, 3);
    tft.print("Identifying...");
  }
  else
  {
    tft.invertDisplay(false);
  }
}

// void tft_write_big_and_blue(const char* text, int chars = 0)
//{
//   // TODO: fill screen with black and then write the word center-middle
//   // aligned as big as you can without causing word wrap using
//   // measureTextWidth and measureTextHeight.
//   // Odds are almost guaranteed that width will be limiting factor.
//
// }

void tft_idle()
{
  if (HW_REV_MATCH(HW_REVISION_0)) // || PID_IS_SECONDARY(NVMEM.pid))
    return;

  //  tft_idle_styleB();
  //}

  // void tft_idle_styleA()
  //{
  //   tft_wakeup();

  //  tft_splash(true); // redraw entire screen (todo: only do this with StyleA?)
  //  tft_tempcontrol(); // draw "OFF" state

  //  tft.fillRect(0, 0, TFT_WIDTH, 20, ILI9341_WHITE);
  //  tft.drawFastHLine(0, 21, TFT_WIDTH, ILI9341_BLACK);
  //  tft.setCursor(3, 3);
  //  tft.setTextSize(2);
  //  if (hw_error)
  //  {
  //    tft.setTextColor(ILI9341_RED);
  //    tft.print("HW: ");
  //    tft.print(max31855.getType(true));
  //    tft.print(" ");
  //    tft.print(max31855.status(true));
  //  }
  //  else
  //  {
  //    tft.setTextColor(ILI9341_BLACK);
  //    tft.print("Idle.");
  //  }
  //}

  // void tft_idle_styleB()
  //{
  //   float pv = temperature;
  //   float sp = l298nhb.getTarget();
  float op = l298nhb_status * l298nhb.getPower();

  if (op != 0)
  {
    // tft_tempcontrol();
    return;
  }

  uint16_t x, y, w, h, pad;

  tft_wakeup();
  tft.fillScreen(ILI9341_BLACK);
  //  tft.setTextColor(QATCH_BLUE_FG);
  //  tft.setTextSize(4);
  //  tft.setFont(Poppins_32_Bold);
  //
  //  String line1 = "nanovisQ";
  //  char buff1[line1.length() + 1]; // trailing NULL
  //  line1.toCharArray(buff1, sizeof(buff1));
  //
  //  w = tft.measureTextWidth(buff1);
  //  h = tft.measureTextHeight(buff1);
  //  x = (TFT_WIDTH - w) / 2;
  //  y = (3 * TFT_HEIGHT / 4) - (h / 2); // middle-bottom
  //
  //  tft.setCursor(x, y);
  //  tft.print(line1);
  //
  //  pad = 5;
  //  x = (TFT_WIDTH - qatch_icon.width) / 2; // center
  //  y = TFT_HEIGHT - (y + h) - pad; // 'y' and 'h' refer to the text placement, not the image // (TFT_HEIGHT / 4) - (logo_image.height / 2); // middle-top

  // TEST: Enable error flags to test LCD error messages
  // hw_error = true;
  // tft_error = true;
  bool transient_error = (max31855.error() != "OK[NONE]");
  if (hw_error || tft_error || transient_error || tft_msgbox || PID_IS_SECONDARY(NVMEM.pid))
  {
    if (tft_msgbox && msgbox_icon == 2)
      tft.setTextColor(ILI9341_GREEN);
    else
      tft.setTextColor(ILI9341_RED);

    tft.setFont(Poppins_16_Bold);

    String line1 = tft_msgbox ? String(msgbox_title) : "Hardware Error Detected"; // hw_error ? "Temp Sensor Error" : "TFT Display Error";
    char buff1[line1.length() + 1];                                               // trailing NULL
    line1.toCharArray(buff1, sizeof(buff1));

    pad = 10;
    w = tft.measureTextWidth(buff1);
    //  h = tft.measureTextHeight(buff1);
    x = (TFT_WIDTH - w) / 2;
    y = pad; // (3 * TFT_HEIGHT / 4) + (h / 2) + pad; // middle-bottom
    tft.setCursor(x, y);
    tft.print(line1);

    // Prep for drawing the icon:
    x = ICON_X;
    y = ICON_Y;

    if (!tft_msgbox || (tft_msgbox && msgbox_icon == 0))
    {
      // Exclamation mark icon:
      tft.fillTriangle(x + 50, y,      // peak
                       x + 15, y + 70, // bottom-left
                       x + 85, y + 70, // bottom-right
                       ILI9341_RED);
      tft.setTextColor(QATCH_GREY_BG);
      tft.setFont(Poppins_32_Bold);
      tft.measureChar('!', &w, &h);
      tft.setCursor(x + 50 - (w / 2), y + 35 - (h / 3));
      tft.print("!");
    }

    if (tft_msgbox && msgbox_icon == 1)
    {
      // Failure circle icon:
      tft.fillCircle(x + 50, y + 35, 35, ILI9341_RED);
      pad = 20;
      for (uint16_t i = pad; i <= 70 - pad; i++)
      {
        w = x + i + 15 - 3;
        h = y + i;
        tft.drawFastHLine(w, h, 7, QATCH_GREY_BG);
        h = y + 70 - i;
        tft.drawFastHLine(w, h, 7, QATCH_GREY_BG);
        if (i == pad || i == 70 - pad)
        {
          h = min(y + i, y + 70 - i); // top cap
          tft.drawFastHLine(w + 1, h - 1, 5, QATCH_GREY_BG);
          tft.drawFastHLine(w + 2, h - 2, 3, QATCH_GREY_BG);
          tft.drawPixel(w + 3, h - 3, QATCH_GREY_BG);
          h = max(y + i, y + 70 - i); // bottom cap
          tft.drawFastHLine(w + 1, h + 1, 5, QATCH_GREY_BG);
          tft.drawFastHLine(w + 2, h + 2, 3, QATCH_GREY_BG);
          tft.drawPixel(w + 3, h + 3, QATCH_GREY_BG);
        }
      }
      // // test pixels:
      // tft.drawPixel(x + 15, y, ILI9341_WHITE);
      // tft.drawPixel(x + 15 + 70, y, ILI9341_WHITE);
      // tft.drawPixel(x + 15, y + 70, ILI9341_WHITE);
      // tft.drawPixel(x + 15 + 70, y + 70, ILI9341_WHITE);
      // tft.drawPixel(x + 50, y + 35, ILI9341_BLACK);
    }

    if (tft_msgbox && msgbox_icon == 2)
      tft.setTextColor(ILI9341_GREEN);
    else
      tft.setTextColor(ILI9341_RED);

    tft.setFont(Poppins_11_Bold);

    //    String line2 = hw_error ? "TEMP SENSOR " + max31855.status(true) : "TFT ERROR: SCREEN MAY FLASH";
    //    if (Serial.dtr()) // only print if port is open in SW (for engineering debug)
    //    {
    //      client->print(line1);
    //      client->print(": ");
    //      client->println(line2);
    //    }

    String line2 = tft_msgbox ? String(msgbox_text) : PID_IS_SECONDARY(NVMEM.pid) ? "PID IS INCORRECT"
                                                                                  : "SERVICE REQUIRED";
    if (l298nhb_auto_off_at != 0)
      line2 = "";                   // in cooldown mode: wait to show msgbox text (title only)
    char buff2[line2.length() + 1]; // trailing NULL
    line2.toCharArray(buff2, sizeof(buff2));

    pad = 80;
    w = tft.measureTextWidth(buff2);
    //  h = tft.measureTextHeight(buff2);
    x = (TFT_WIDTH - w) / 2;
    y = ICON_Y + pad; // (3 * TFT_HEIGHT / 4) + (h / 2) + pad; // middle-bottom
    tft.setCursor(x, y);
    tft.print(line2);

    //    tft.print("HW: ");
    //    tft.print(max31855.getType(true));
    //    tft.print(" ");
    //    tft.print(max31855.status(true));
  }
  else
  {
    x = ICON_X;
    y = ICON_Y;
    tft.writeRect(x, y, qatch_icon.width, qatch_icon.height, (uint16_t *)(qatch_icon.pixel_data));
  }

  x = TEXT_X;
  y = TEXT_Y;
  tft.writeRect(x, y, nanovisQ_black.width, nanovisQ_black.height, (uint16_t *)(nanovisQ_black.pixel_data));

  tft.setTextColor(QATCH_BLUE_FG);
  tft.setFont(Poppins_16_Bold);

  String line2 = "ID: [????????]";
  char buff2[line2.length() + 1]; // trailing NULL
  //  line2.toCharArray(buff2, sizeof(buff2));
  sprintf(buff2, "ID: %lu", teensyUsbSN());

  pad = 10;
  w = tft.measureTextWidth(buff2);
  //  h = tft.measureTextHeight(buff2);
  x = (TFT_WIDTH - w) / 2;
  //  y = (3 * TFT_HEIGHT / 4) + (h / 2) + pad; // middle-bottom
  y = TEXT_Y + nanovisQ_black.height + pad;

  tft.setCursor(x, y);
  tft.print(buff2);

  if (tft_msgbox && msgbox_icon == 2)
  {
    x = ICON_X;
    y = ICON_Y;

    // Success circle icon:
    tft.fillCircle(x + 50, y + 35, 35, ILI9341_GREEN);
    pad = 10;
    for (uint16_t i = pad; i <= 70 - 2 * pad; i++)
    {
      delay(25);
      w = x + i + 15 + 2;
      h = (i < 22 ? y + 35 + i : y + 79 - i) - pad + 2;
      // Serial.printf("(%u, %u) (%u, %u)\n", x + 50, y + 35, w, h);
      tft.drawFastHLine(w, h, 7, QATCH_GREY_BG);
      if (i == pad || i == 70 - 2 * pad)
      {
        tft.drawFastHLine(w + 1, h - 1, 5, QATCH_GREY_BG);
        tft.drawFastHLine(w + 2, h - 2, 3, QATCH_GREY_BG);
        tft.drawPixel(w + 3, h - 3, QATCH_GREY_BG);
      }
      if (i == 22)
      {
        tft.drawFastHLine(w + 1, h + 1, 5, QATCH_GREY_BG);
        tft.drawFastHLine(w + 2, h + 2, 3, QATCH_GREY_BG);
        tft.drawPixel(w + 3, h + 3, QATCH_GREY_BG);
      }
    }
  }
}

// void tft_testmode()
//{
//   uint16_t x, y, w, h, pad;
//
//   tft_wakeup();
//   tft.fillRect(0, 0, TFT_WIDTH, 20, ILI9341_WHITE);
//   tft.drawFastHLine(0, 21, TFT_WIDTH, ILI9341_BLACK);
//   tft.setCursor(3, 3);
//   tft.setTextSize(2);
//   tft.setTextColor(ILI9341_RED);
//   tft.print("Test Mode!");
//
//   while (true) {
//     Serial.begin(BAUD);
//
//     tft_splash(true);
//
//     while (!Serial.available());
//     if (Serial.available())
//     {
//       while (Serial.available() > 0) { // flush buffer
//         char t = Serial.read();
//       }
//       //      break;
//     }
//
//     tft_wakeup();
//     tft.fillScreen(QATCH_GREY_BG);
//     //    tft.setTextColor(QATCH_BLUE_FG);
//     //    tft.setFont(Poppins_40_Bold);
//
//     tft.setTextColor(ILI9341_BLACK);
//     tft.setFontAdafruit(); // default console font
//     tft.setTextSize(1);
//
//     //    String line0 = "nanovisQ";
//     //    char buff0[line0.length() + 1]; // trailing NULL
//     //    line0.toCharArray(buff0, sizeof(buff0));
//     //
//     //    w = tft.measureTextWidth(buff0);
//     //    h = tft.measureTextHeight(buff0);
//     //    x = (TFT_WIDTH - w) / 2;
//     //    y = (3 * TFT_HEIGHT / 4) - (h / 2) - 18; // middle-bottom
//     //
//     //    tft.setCursor(x, y);
//     //    tft.print(line0);
//
//     pad = 35;
//     x = (TFT_WIDTH - nanovisQ_grey.width) / 2; // center
//     //    y = TFT_HEIGHT - (y + h) - pad; // 'y' and 'h' refer to the text placement, not the image // (TFT_HEIGHT / 4) - (logo_image.height / 2); // middle-top
//     y = (3 * TFT_HEIGHT / 4) - (h / 2) - 25; // middle-bottom
//
//     tft.writeRect(x, y, nanovisQ_grey.width, nanovisQ_grey.height, (uint16_t*)(nanovisQ_grey.pixel_data));
//     tft.setCursor(x, y);
//     tft.printf("X=%03u,Y=%03u", x, y);
//
//
//     pad = 43;
//     x = (TFT_WIDTH - splash_icon.width) / 2; // center
//     y = TFT_HEIGHT - (y + h) - pad; // 'y' and 'h' refer to the text placement, not the image // (TFT_HEIGHT / 4) - (logo_image.height / 2); // middle-top
//
//     tft.writeRect(x, y, splash_icon.width, splash_icon.height, (uint16_t*)(splash_icon.pixel_data));
//     tft.setCursor(x, y);
//     tft.printf("X=%03u,Y=%03u", x, y);
//
//     tft.setTextColor(ILI9341_BLACK);
//     tft.setFontAdafruit(); // default console font
//     tft.setTextSize(2);
//     //    tft.writeRect(0, 0, boot_splash.width, boot_splash.height, (uint16_t*)(boot_splash.pixel_data));
//     h = 20; // line height
//     pad = 5;
//
//     String line1 = CODE_VERSION; // : "UPDATING";
//     char buff1[line1.length() + 1]; // trailing NULL
//     line1.toCharArray(buff1, sizeof(buff1));
//     w = tft.measureTextWidth(buff1);
//     tft.setCursor((TFT_WIDTH - w) / 2, TFT_HEIGHT - (2 * h) - pad);
//     tft.print(line1);
//
//     String line2 = RELEASE_DATE; // : "FIRMWARE";
//     char buff2[line2.length() + 1]; // trailing NULL
//     line2.toCharArray(buff2, sizeof(buff2));
//     w = tft.measureTextWidth(buff2);
//     tft.setCursor((TFT_WIDTH - w) / 2, TFT_HEIGHT - (1 * h) - pad);
//     tft.print(line2);
//
//     //    for (uint16_t ty = y; ty <= y + qatch_icon.height; ty++)
//     //    {
//     //      for (uint16_t tx = x; tx <= x + qatch_icon.width; tx++)
//     //      {
//     //        if (tft.readPixel(tx, ty) == ILI9341_BLACK)
//     //          tft.drawPixel(tx, ty, QATCH_GREY_BG);
//     //      }
//     //    }
//
//     while (!Serial.available());
//     if (Serial.available())
//     {
//       while (Serial.available() > 0) { // flush buffer
//         char t = Serial.read();
//       }
//       //      break;
//     }
//   }
// }

// void tft_tempbusy()
//{
//   tft_tempbusy_styleB();
// }

// void tft_tempbusy_styleA()
//{
//   tft_wakeup();

//  int rect_x = TFT_WIDTH / 2;
//  int rect_y = 69;
//  int rect_w = 203;
//  int rect_h = 24;
//  int rect_r = 15;
//  int pad = 2;

//  if (tft.readPixel(rect_x, rect_y + (rect_h / 2)) != ILI9341_BLACK)
//  {
//    Serial.println("Drawing TEC scale...");
//    tft.fillRect(0, 22, TFT_WIDTH, 115, QATCH_GREY_BG); // only do once, not on every bar update
//    tft.fillRoundRect(rect_x - 101 - 2 * pad, rect_y - 2 * pad, rect_w + 4 * pad, rect_h + 4 * pad, rect_r, ILI9341_BLACK);

//    tft.fillRoundRect(rect_x - 101 - pad, rect_y - pad, rect_w + 2 * pad, rect_h + 2 * pad, rect_r, ILI9341_WHITE);
//    tft.drawFastVLine(rect_x - 1, rect_y - pad, rect_h + 2 * pad, ILI9341_BLACK);
//    tft.drawFastVLine(rect_x, rect_y - pad, rect_h + 2 * pad, ILI9341_BLACK);
//    tft.drawFastVLine(rect_x + 1, rect_y - pad, rect_h + 2 * pad, ILI9341_BLACK);
//  }

//  tft.setTextColor(ILI9341_BLACK);
//  tft.setTextSize(1);
//  tft.fillRect(0, rect_y + rect_h + 5 * pad, TFT_WIDTH, rect_h, QATCH_GREY_BG);
//  char buf[32]; // trailing NULL
//  if (!l298nhb.active())
//  {
//    sprintf(buf, "Temp Control is idle.");
//  }
//  else
//  {
//    sprintf(buf, "Temp Control running in background.");
//  }
//  uint16_t w = tft.measureTextWidth(buf);
//  tft.setCursor((TFT_WIDTH - w) / 2, rect_y + rect_h + 5 * pad);
//  tft.print(buf);
//}

// void tft_tempbusy_styleB()
//{
//  do nothing
// }

void tft_cooldown_start()
{
  // Serial.println("TFT Cooldown started");
  l298nhb_auto_off_at = millis() + L298NHB_COOLDOWN; // calculate time to idle
  // special case in event of rollover in prior line math
  if (l298nhb_auto_off_at == 0) 
  {
    l298nhb_auto_off_at = 1;
  }
  tft_cooldown();                                    // will also call _prepare() when drawing full cooldown UI
}

void tft_cooldown_prepare()
{
  // prepare cooldown variables init
  // const short num_ring_steps = 4;                           // outer ring is broken into four segments
  // short total_time = L298NHB_COOLDOWN / 1000;               // total countdown duration, in seconds
  last_pct = (((l298nhb_auto_off_at - millis())                // total seconds for cooldown, decremented, aligned with task timer
             - (l298nhb_auto_off_at % 1000)
             + (l298nhb_task_timer % 1000)) / 1000) + 1;
  // last_pct = max(0, min(total_time, last_pct));             // bound within range of 0 to 'total_time'
  // last_op = floor(total_time / num_ring_steps);             // seconds between ring steps, constant
  // last_sp = num_ring_steps - floor(last_pct / last_op) - 1; // ring step counter, incremented
  // last_sp = max(0, min(num_ring_steps, last_sp));           // bound within range of 0 to 'num_ring_steps'
  // Serial.printf("last_pct=%u, last_op=%1.0f, last_sp=%1.0f\n", last_pct, last_op, last_sp);
}

void tft_drawCircleHelper(int16_t x0, int16_t y0,
                          int16_t r, uint8_t cornername, uint16_t color)
{
  int16_t f = 1 - r;
  int16_t ddF_x = 1;
  int16_t ddF_y = -2 * r;
  int16_t x = 0;
  int16_t y = r;
  int px_drawn = 0;
  int xold;

  xold = x;
  while (x < y)
  {
    if (f >= 0)
    {
      y--;
      ddF_y += 2;
      f += ddF_y;
    }
    x++;
    ddF_x += 2;
    f += ddF_x;
    if (f >= 0 || x == y)
    { // time to draw the new line segment
      if (cornername & 0x4)
      {
        tft.drawFastHLine(x0 + xold + 1, y0 + y, x - xold, color);
        tft.drawFastVLine(x0 + y, y0 + xold + 1, x - xold, color);
        px_drawn += 2 * (x - xold);
      }
      if (cornername & 0x2)
      {
        tft.drawFastHLine(x0 + xold + 1, y0 - y, x - xold, color);
        tft.drawFastVLine(x0 + y, y0 - x, x - xold, color);
        px_drawn += 2 * (x - xold);
      }
      if (cornername & 0x8)
      {
        tft.drawFastVLine(x0 - y, y0 + xold + 1, x - xold, color);
        tft.drawFastHLine(x0 - x, y0 + y, x - xold, color);
        px_drawn += 2 * (x - xold);
      }
      if (cornername & 0x1)
      {
        tft.drawFastVLine(x0 - y, y0 - x, x - xold, color);
        tft.drawFastHLine(x0 - x, y0 - y, x - xold, color);
        px_drawn += 2 * (x - xold);
      }
      xold = x;
    } // draw new line segment
  }

  Serial.printf("pixels drawn = %u\n", px_drawn);
}

void tft_progress_show(float target_pct)
{
  // tft_cooldown();

  int mid_offset = 50; // offset to center from top-left corner
  int mid_radius = 46; // center radius of progress circle
  int band_width = 3;  // radial width of band (from 43 to 49)
  int num_pts = 360;   // number of pixels in outer circumference

  // tft_drawCircleHelper(ICON_X + mid_offset, ICON_Y + mid_offset, mid_radius, 0xF, ILI9341_MAGENTA);

  // for (target_pct = 100; target_pct >= 0; target_pct--)
  if (true)
  {

    for (int i = 0; i < num_pts; i++)
    {
      int x0 = ICON_X + mid_offset;
      int y0 = ICON_Y + mid_offset;

      int deg = (360 * i) / num_pts;
      float rad = radians(deg);
      float pct = deg / 3.6;

      int x_offset;
      int y_offset;
      uint16_t color;

      for (int r = mid_radius - band_width + 1; r <= mid_radius + band_width; r++)
      {
        x_offset = round(r * +sin(rad));
        y_offset = round(r * -cos(rad));

        int x = x0 + x_offset;
        int y = y0 + y_offset;
        color = (pct < target_pct || (x == x0 && y < y0)) ? ILI9341_BLACK : QATCH_GREY_BG;

        tft.drawPixel(x, y, color);

        if (r == mid_radius - band_width + 1 || abs(x - x0) <= 1 || abs(pct - target_pct) > 1 || tft_error)
          continue; // skip advanced pixel corrections of static

        // ADVANCED: detect and correct static pixels
        if (tft.readPixel(x > x0 ? x - 1 : x + 1, y) != color)
        {
          // tft.drawPixel(x > x0 ? x - 1 : x + 1, y, color);
          if (tft.readPixel(x > x0 ? x - 2 : x + 2, y) == color || tft.readPixel(x > x0 ? x - 1 : x + 1, y > y0 ? y - 1 : y + 1) == color)
            // only draw if pixel left/right or below/above wrong pixel is the correct color
            tft.drawPixel(x > x0 ? x - 1 : x + 1, y, color);
        }
      }

      // Serial.printf("i=%u, deg=%u, rad=%f, pct=%f, xo=%i, yo=%i, c=%u\n", i, deg, rad, pct, x_offset, y_offset, color);
    }

    // delay(1000);
  }

  // delay(1000 * 30);
}

#if false
void tft_cooldown_drawRing(int step, uint16_t color)
{
  return; // do not process this function

  // fill the outer bands at quarter increments
  // start with black fill, empty to grey fill
  const short r1 = 50;
  const short r2 = 42;

  for (int i = 1; i < (r1 - r2); i++) // from outer white to inner black, retaining outer and inner circle colors
  {
    const short r = r1 - i;
    const short w = 100 - 2 * i;
    const short h = 100 - 2 * i;
    const short x = ICON_X + i;
    const short y = ICON_Y + i;

    Serial.printf("r=%u, w=%u, h=%u, x=%u, y=%u\n", r, w, h, x, y);
    switch (step)
    {
    case 1:
      tft_drawCircleHelper(x + r, y + r, r, 1, color); // 1: 75-100%
      break;
    case 2:
      tft_drawCircleHelper(x + r, y + h - r, r, 8, color); // 8: 50-75%
      break;
    case 3:
      tft_drawCircleHelper(x + w - r, y + h - r, r, 4, color); // 4: 25-50%
      break;
    case 4:
      tft_drawCircleHelper(x + w - r, y + r, r, 2, color); // 2: 0-25%
      break;
    default:
      Serial.printf("Unknown step: %u\n", step);
      return; // do not continue looping
    }
  }
}
#endif

void tft_cooldown()
{
  if (HW_REV_MATCH(HW_REVISION_0) || PID_IS_SECONDARY(NVMEM.pid))
    return;

  tft_wakeup();

  // // testing, force redraw
  // tft.fillScreen(ILI9341_BLACK);

  // do we need to redraw the full screen?
  // target pixel 1: center of the 'i' dot in 'nanovisQ'
  // target pixel 2: top center of icon, grey outer circle border
  if (tft.readPixel(TEXT_X + 178, TEXT_Y + 5) != QATCH_BLUE_FG ||
      tft.readPixel(ICON_X + 50, ICON_Y) != QATCH_GREY_BG)
  {
    // (re)calculate UI progress and init vars on "cooldown" start/resume
    tft_cooldown_prepare();

    tft_idle();

    const short icon_wh = 100;
    const short icon_xy = 50;
    const short icon_r1 = 50;
    const short icon_r2 = 42;

    tft.fillCircle(ICON_X + icon_xy, ICON_Y + icon_xy, icon_r1, QATCH_GREY_BG);
    tft.fillCircle(ICON_X + icon_xy, ICON_Y + icon_xy, icon_r2, QATCH_BLUE_FG);

    // tft.fillCircle(ICON_X + icon_xy, ICON_Y + icon_xy, icon_r1 - 1, QATCH_BLUE_FG);
    // tft.drawCircle(ICON_X + icon_xy, ICON_Y + icon_xy, icon_r1, QATCH_GREY_BG);
    // tft.drawCircle(ICON_X + icon_xy, ICON_Y + icon_xy, icon_r2, QATCH_BLUE_FG);
    // // tft.drawCircle(ICON_X + icon_xy, ICON_Y + icon_xy, icon_r2, QATCH_GREY_BG);

    // for (int i = 1; i <= (icon_r1 - icon_r2); i++)
    // {
    //   tft.drawCircle(ICON_X + icon_xy, ICON_Y + icon_xy, icon_r2 + i, QATCH_GREY_BG);
    // }

    // draw 4 compass marks on outer ring segments
    // tft.drawFastHLine(ICON_X + 1, ICON_Y + icon_xy, icon_r1 - icon_r2 - 1, ILI9341_BLACK);                                 // west
    // tft.drawFastHLine(ICON_X + icon_wh - (icon_r1 - icon_r2) + 1, ICON_Y + icon_xy, icon_r1 - icon_r2 - 1, ILI9341_BLACK); // east
    // tft.drawFastVLine(ICON_X + icon_xy, ICON_Y + 1, icon_r1 - icon_r2 - 1, ILI9341_BLACK);                                 // north
    // tft.drawFastVLine(ICON_X + icon_xy, ICON_Y + icon_wh - (icon_r1 - icon_r2) + 1, icon_r1 - icon_r2 - 1, ILI9341_BLACK); // south

    tft.setTextColor(ILI9341_BLACK);
    tft.setFont(Poppins_16_Bold);

    sprintf(last_line_label, "-:--");
    short tw = tft.measureTextWidth(last_line_label);
    short th = tft.measureTextHeight(last_line_label);

    // draw number placeholder in circle
    tft.setCursor(ICON_X + (100 - tw) / 2, ICON_Y + 2 * (100 - th) / 5);
    tft.print(last_line_label);

    tft.setTextColor(ILI9341_BLACK);
    tft.setFont(Poppins_12_Bold);

    sprintf(last_line_label, "venting");
    tw = tft.measureTextWidth(last_line_label);
    th = tft.measureTextHeight(last_line_label);

    tft.setCursor(ICON_X + (icon_wh - tw) / 2, ICON_Y + 2 * (icon_wh - th) / 3);
    tft.print(last_line_label);

    // redraw current ring state
    for (int i = 1; i < 100 - (float)(100.0 * 1000.0 * last_pct / L298NHB_COOLDOWN); i++)
    {
      tft_progress_show(i);
    }
    // for (int i = 1; i <= (int)last_sp; i++)
    // {
    //   // Serial.printf("Re-draw quarter ring %u.\n", i);
    //   tft_cooldown_drawRing(i, QATCH_BLUE_FG);
    // }
  }
  // else
  // {
  // // clear last number in circle
  // tft.fillRect(ICON_X + 20, ICON_Y + 25, 60, 30, QATCH_BLUE_FG);

  // if ((last_pct % (int)last_op) == 0)
  // {
  //   // Serial.printf("Quarter increment %1.0f/%u @ %u sec remaining\n", last_sp + 1, 4, last_pct);
  //   tft_cooldown_drawRing((int)last_sp + 1, QATCH_BLUE_FG);
  //   last_sp += 1; // float increment by 1
  // }
  // else
  // {
  //   uint16_t color;
  //   if ((last_pct % 2) == 0)
  //   {
  //     color = QATCH_GREY_BG;
  //   }
  //   else
  //   {
  //     color = QATCH_BLUE_FG;
  //   }
  //   tft_cooldown_drawRing((int)last_sp + 1, color); // blink next ring segment to show as in-progress
  // }
  // }

  if (L298NHB_COOLDOWN / 1000 >= 100)
  {
    tft_progress_show(100 - (float)(100.0 * 1000.0 * last_pct / L298NHB_COOLDOWN));
  }
  else
  {
    int last = 100 - (float)(100.0 * 1000.0 * (last_pct + 1) / L298NHB_COOLDOWN);
    int this_pct = 100 - (float)(100.0 * 1000.0 * last_pct / L298NHB_COOLDOWN);
    for (int i = last; i < this_pct; i++)
    {
      tft_progress_show(i);
    }
  }

  short rem_t = max(0, min(L298NHB_COOLDOWN / 1000, last_pct));
  short rem_m = floor(rem_t / 60);
  short rem_s = (rem_t % 60);

  if (last_pct > 0)
    last_pct--; // stop at zero

  tft.setTextColor(ILI9341_BLACK);
  tft.setFont(Poppins_16_Bold);

  sprintf(last_line_label, "%1u:%02u", rem_m, rem_s);
  short tw = tft.measureTextWidth(last_line_label);
  short th = tft.measureTextHeight(last_line_label);

  // clear last number in circle
  tft.fillRect(ICON_X + 20, ICON_Y + 25, 60, 30, QATCH_BLUE_FG);

  // draw new number in circle
  tft.setCursor(ICON_X + (100 - tw) / 2, ICON_Y + 2 * (100 - th) / 5);
  tft.print(last_line_label);

  // // debug only:
  // tft.setTextColor(QATCH_BLUE_FG);
  // tft.setFont(Poppins_10_Bold);
  // tft.setFontAdafruit();
  // tft.fillRect(0, 0, 100, 25, ILI9341_BLACK);
  // tft.setCursor(0, 0);
  // tft.printf("%u", rem_t);

  // tft.drawPixel(TEXT_X + 178, TEXT_Y + 5, ILI9341_RED);
}

void tft_tempcontrol()
{
  if (HW_REV_MATCH(HW_REVISION_0) || PID_IS_SECONDARY(NVMEM.pid))
    return;

  //  tft_tempcontrol_styleB();
  //}

  // void tft_tempcontrol_styleA()
  //{
  //   tft_wakeup();
  //   tft.fillRect(0, 0, TFT_WIDTH, 20, ILI9341_WHITE);
  //   tft.drawFastHLine(0, 21, TFT_WIDTH, ILI9341_BLACK);
  //   tft.setCursor(3, 3);
  //   tft.setTextSize(2);
  //   tft.setTextColor(ILI9341_BLACK);
  //   tft.print("Temp Control running...");

  //  int rect_x = TFT_WIDTH / 2;
  //  int rect_y = 69;
  //  int rect_w = 203;
  //  int rect_h = 24;
  //  int rect_r = 15;
  //  int pad = 2;

  //  if (tft.readPixel(rect_x, rect_y + (rect_h / 2)) != ILI9341_BLACK)
  //  {
  //    Serial.println("Drawing TEC scale...");
  //    tft.fillRect(0, 22, TFT_WIDTH, 115, QATCH_GREY_BG); // only do once, not on every bar update
  //    tft.fillRoundRect(rect_x - 101 - 2 * pad, rect_y - 2 * pad, rect_w + 4 * pad, rect_h + 4 * pad, rect_r, ILI9341_BLACK);
  //  }

  //  float pv = temperature;
  //  float sp = l298nhb.getTarget();
  //  float op = l298nhb_status * l298nhb.getPower();

  //  byte pct;
  //  if (op == 0) pct = 0;
  //  else if (op > 0) pct = (100 * op) / 150;
  //  else pct = (100 * op) / -255;

  //  tft.fillRoundRect(rect_x - 101 - pad, rect_y - pad, rect_w + 2 * pad, rect_h + 2 * pad, rect_r, ILI9341_WHITE);
  //  tft.drawFastVLine(rect_x - 1, rect_y - pad, rect_h + 2 * pad, ILI9341_BLACK);
  //  tft.drawFastVLine(rect_x, rect_y - pad, rect_h + 2 * pad, ILI9341_BLACK);
  //  tft.drawFastVLine(rect_x + 1, rect_y - pad, rect_h + 2 * pad, ILI9341_BLACK);

  //  tft.setTextColor(ILI9341_BLACK);
  //  tft.setTextSize(1);
  //  tft.fillRect(0, rect_y + rect_h + 5 * pad, TFT_WIDTH, rect_h, QATCH_GREY_BG);
  //  char buf[32]; // trailing NULL
  //  if (op == 0 || !l298nhb.active())
  //  {
  //    sprintf(buf, "PV:%sC SP:%2.2fC OP:%s", "--.--", sp, "[OFF]");
  //    sprintf(buf, "Temp Control is idle.");
  //  }
  //  else
  //  {
  //    sprintf(buf, "PV:%2.2fC SP:%2.2fC OP:%+04.0f", pv, sp, op);
  //  }
  //  uint16_t w = tft.measureTextWidth(buf);
  //  tft.setCursor((TFT_WIDTH - w) / 2, rect_y + rect_h + 5 * pad);
  //  tft.print(buf);

  //  if (op != 0)
  //  {
  //    if (pct == 0) pct = 1;
  //    if (op > 0)
  //    {
  // heating
  //      if (pct < 100 - rect_r)
  //      {
  //        tft.fillRect(rect_x + 1, rect_y, pct + 1, rect_h, ILI9341_RED);
  //      }
  //      else
  //      {
  //        int x = rect_x + 1;
  //        int y = rect_y;
  //        int w = pct + 1;
  //        int h = rect_h;
  //        int r = rect_r - (100 - pct); // min(pct-1, rect_r);
  //        tft.fillRect(x, y, w - r, h, ILI9341_RED);
  //        tft.fillCircleHelper(x + w - r - 1, y + r, r, 1, h - 2 * r - 1, ILI9341_RED);
  // fillRect(x+r, y, w-2*r, h, color); --> fillRect(x, y, w-r, h, color);
  // fillCircleHelper(x+w-r-1, y+r, r, 1, h-2*r-1, color);
  //      }
  //    }
  //    else
  //    {
  // cooling
  //      if (pct < 100 - rect_r)
  //      {
  //        tft.fillRect(rect_x - (pct + 1), rect_y, pct + 1, rect_h, ILI9341_BLUE);
  //      }
  //      else
  //      {
  //        int x = rect_x - (pct + 1);
  //        int y = rect_y;
  //        int w = pct + 1;
  //        int h = rect_h;
  //        int r = rect_r - (100 - pct); // min(pct-1, rect_r);
  //        tft.fillRect(x + r, y, w - r, h, ILI9341_BLUE);
  //        tft.fillCircleHelper(x + r    , y + r, r, 2, h - 2 * r - 1, ILI9341_BLUE);
  // fillRect(x+r, y, w-r, h, color);
  // fillCircleHelper(x+r    , y+r, r, 2, h-2*r-1, color);
  //      }
  //    }
  //  }

  //  bool dir = true;
  //  bool do_once = true;
  //  int value = -255;
  //  while (true) {
  //    Serial.begin(BAUD);
  //    if (Serial.available())
  //    {
  //      while (Serial.available() > 0) { // flush buffer
  //        char t = Serial.read();
  //      }
  //      break;
  //    }
  //    //    String message_str = Serial.readStringUntil('\n');
  //    //    int value = message_str.toInt(); // -255 to 150, inclusive
  //    delay(100);
  //    if (value >= 150) dir = false;
  //    if (value <= -255) dir = true;
  //    value += (dir ? 1 : -1);
  //    byte pct;
  //    if (value == 0) pct = 0;
  //    else if (value > 0) pct = (100 * value) / 150;
  //    else pct = (100 * value) / -255;
  //    Serial.print(value);
  //    Serial.print(" -> ");
  //    Serial.println(pct);
  //
  //    int rect_x = TFT_WIDTH / 2;
  //    int rect_y = 69;
  //    int rect_w = 203;
  //    int rect_h = 24;
  //    int rect_r = 15;
  //    int pad = 2;
  //    if (do_once)
  //    {
  //      do_once = false;
  //      tft.fillRoundRect(rect_x - 101 - 2 * pad, rect_y - 2 * pad, rect_w + 4 * pad, rect_h + 4 * pad, rect_r, ILI9341_BLACK);
  //    }
  //    tft.fillRoundRect(rect_x - 101 - pad, rect_y - pad, rect_w + 2 * pad, rect_h + 2 * pad, rect_r, ILI9341_WHITE);
  //    tft.drawFastVLine(rect_x - 1, rect_y - pad, rect_h + 2 * pad, ILI9341_BLACK);
  //    tft.drawFastVLine(rect_x, rect_y - pad, rect_h + 2 * pad, ILI9341_BLACK);
  //    tft.drawFastVLine(rect_x + 1, rect_y - pad, rect_h + 2 * pad, ILI9341_BLACK);
  //
  //    tft.setTextColor(ILI9341_BLACK);
  //    tft.setTextSize(1);
  //    tft.fillRect(rect_x - 101, rect_y + rect_h + 5 * pad, rect_w, rect_h, QATCH_BGCOLOR);
  //    char buf[32]; // trailing NULL
  //    float temp = 25 + (random(-100, 100) / 100.0);
  //    Serial.println(temp);
  //    if (value == 0 || !l298nhb.active())
  //    {
  //      sprintf(buf, "PV:%2.2fC SP:%2.2fC OP:%s", NAN, temp, "[OFF]");
  //    }
  //    sprintf(buf, "PV:%2.2fC SP:%2.2fC OP:%+04.0f", NAN, temp, (float)value);
  //    // String tec_label = "PV:--.--C SP:21.00C OP:[OFF]"; // now with null-terminator
  //    uint16_t w = tft.measureTextWidth(buf);
  //    tft.setCursor((TFT_WIDTH - w) / 2, rect_y + rect_h + 5 * pad);
  //    tft.print(buf);
  //
  //    if (value != 0)
  //    {
  //      if (pct == 0) pct = 1;
  //      if (value > 0)
  //      {
  //        // heating
  //        if (pct < 100 - rect_r)
  //        {
  //          tft.fillRect(rect_x + 1, rect_y, pct + 1, rect_h, ILI9341_RED);
  //        }
  //        else
  //        {
  //          int x = rect_x + 1;
  //          int y = rect_y;
  //          int w = pct + 1;
  //          int h = rect_h;
  //          int r = rect_r - (100 - pct); // min(pct-1, rect_r);
  //          tft.fillRect(x, y, w - r, h, ILI9341_RED);
  //          tft.fillCircleHelper(x + w - r - 1, y + r, r, 1, h - 2 * r - 1, ILI9341_RED);
  //          //fillRect(x+r, y, w-2*r, h, color); --> fillRect(x, y, w-r, h, color);
  //          //fillCircleHelper(x+w-r-1, y+r, r, 1, h-2*r-1, color);
  //        }
  //      }
  //      else
  //      {
  //        // cooling
  //        if (pct < 100 - rect_r)
  //        {
  //          tft.fillRect(rect_x - (pct + 1), rect_y, pct + 1, rect_h, ILI9341_BLUE);
  //        }
  //        else
  //        {
  //          int x = rect_x - (pct + 1);
  //          int y = rect_y;
  //          int w = pct + 1;
  //          int h = rect_h;
  //          int r = rect_r - (100 - pct); // min(pct-1, rect_r);
  //          tft.fillRect(x + r, y, w - r, h, ILI9341_BLUE);
  //          tft.fillCircleHelper(x + r    , y + r, r, 2, h - 2 * r - 1, ILI9341_BLUE);
  //          //fillRect(x+r, y, w-r, h, color);
  //          //fillCircleHelper(x+r    , y+r, r, 2, h-2*r-1, color);
  //        }
  //      }
  //    }

  //    for (int r = 0; r <= rect_r; r++)
  //    {
  //      tft.drawRoundRect(rect_x - 101 - r, rect_y - r, rect_w + 2 * r, rect_h + 2 * r, r, ILI9341_GREEN);
  //    }
  //  }
  //}

  // void tft_tempcontrol_styleB()
  //{
  //   while (true)
  //   {
  //     for (int test_i = -255; test_i <= 150; test_i++)
  //     {
  //       delay(1000);
  //   if (test_i == 0) test_i++; // skip zero

  float pv = round(temperature / TEMP_RESOLUTION) * TEMP_RESOLUTION; // https://arduino.stackexchange.com/a/28469
  float sp = l298nhb.getTarget();
  float op = l298nhb_status * l298nhb.getPower(); // test_i;
  uint16_t fillColor = ILI9341_BLACK;             // change to test positioning of rects

  if (op == 0)
  {
    // tft_idle();
    return;
  }

  tft_wakeup();
  //  tft.fillScreen(ILI9341_BLACK);

  int rect_x = TFT_WIDTH / 2;
  int rect_y = 50;
  int rect_h = 24;
  int rect_w = 203 + rect_h;
  int rect_r; // half of height
  int pad = 3;

  tft.setFont(Poppins_10_Bold);
  tft.setTextColor(QATCH_BLUE_FG);

  char buf[42]; // trailing NULL
  sprintf(buf, "PV: --.--C    SP: --.--C    OP: ----");
  uint16_t w = tft.measureTextWidth(buf);
  uint16_t h = tft.measureTextHeight(buf);

  if (tft.readPixel(rect_x / 2, rect_y - 2 * pad) != QATCH_BLUE_FG)
  {
    //    Serial.println("Drawing TEC scale...");
    last_temp_label = 255;             // none
    last_line_label[0] = '\0';         // invalidate string
    last_pv = last_sp = last_op = 255; // impossible values
    last_pct = 0;

    tft.fillScreen(ILI9341_BLACK);
    //    tft.fillRect(0, 22, TFT_WIDTH, 115, fillColor); // only do once, not on every bar update
    rect_r = (rect_h + 4 * pad) / 2;
    tft.fillRoundRect(rect_x - 113 - 2 * pad, rect_y - 2 * pad, rect_w + 4 * pad, rect_h + 4 * pad, rect_r, QATCH_BLUE_FG);

    tft.setCursor((TFT_WIDTH - w) / 2, rect_y + rect_h + 5 * pad);
    tft.print(buf);

    //    tft.fillRoundRect(rect_x - 101 - pad, rect_y - pad, rect_w + 2 * pad, rect_h + 2 * pad, rect_r, ILI9341_WHITE);
    //    tft.drawFastVLine(rect_x - 1, rect_y - pad, rect_h + 2 * pad, QATCH_BLUE_FG);
    //    tft.drawFastVLine(rect_x, rect_y - pad, rect_h + 2 * pad, QATCH_BLUE_FG);
    //    tft.drawFastVLine(rect_x + 1, rect_y - pad, rect_h + 2 * pad, QATCH_BLUE_FG);
  }

  short pct;
  if (op == 0)
    pct = 0;
  else if (op > 0)
    pct = max(1, (100 * op) / MAX_PWR_HEAT); // 0 -> 1
  else
    pct = min(-1, (100 * op) / MAX_PWR_COOL); // 0 -> -1

  if (pct != last_pct)
  {
    rect_r = (rect_h + 2 * pad) / 2;
    tft.fillRoundRect(rect_x - 113 - pad, rect_y - pad, rect_w + 2 * pad, rect_h + 2 * pad, rect_r, QATCH_GREY_BG);
    tft.drawFastVLine(rect_x - 1, rect_y - pad, rect_h + 2 * pad, QATCH_BLUE_FG);
    tft.drawFastVLine(rect_x, rect_y - pad, rect_h + 2 * pad, QATCH_BLUE_FG);
    tft.drawFastVLine(rect_x + 1, rect_y - pad, rect_h + 2 * pad, QATCH_BLUE_FG);
  }

  //  tft.setFontAdafruit(); // default console font
  //  tft.setFont(Poppins_10_Bold);
  //  tft.setTextColor(QATCH_BLUE_FG);
  //  tft.setTextSize(1);

  bool rewrite_PV_color = false;
  if (max31855.status() != 0)
  {
    tft.setTextColor(ILI9341_RED);
    tft.setFont(Poppins_16_Bold);

    String line0 = "Hardware Error Detected";
    char buff0[line0.length() + 1]; // trailing NULL
    line0.toCharArray(buff0, sizeof(buff0));

    uint16_t _pad = 10;
    uint16_t _w = tft.measureTextWidth(buff0);
    // uint16_t _h = tft.measureTextHeight(buff0);
    uint16_t _x = (TFT_WIDTH - _w) / 2;
    uint16_t _y = _pad;
    //    Serial.printf("%u;%u;%u;%u", _x, _y, _w, _h);
    tft.setCursor(_x, _y);
    tft.print(line0);

    hw_error = true;

    tft.setFont(Poppins_10_Bold);
    tft.setTextColor(QATCH_BLUE_FG);
  }
  else if (hw_error)
  {
    // persist hw error message on LCD
    //    tft.fillRect(19, 10, 282, 16, fillColor);
    rewrite_PV_color = true;
    hw_error = false;
  }

  int x = (TFT_WIDTH - w) / 2;
  int y = rect_y + rect_h + 5 * pad;
  int x_pv = 75;  // x + tft.measureTextWidth(buf, 3);
  int x_sp = 154; // x + tft.measureTextWidth(buf, 15);
  int x_op = 235; // x + tft.measureTextWidth(buf, 28);
  int tw = 50;    // tft.measureTextWidth(buf, 10) - x_pv;
  h = 10;         // matches font size
  //  uint16_t th;
  //  tft.measureChar('P', &w, &th);

  //  Serial.println(x_pv);

  if (pv != last_pv || hw_error || rewrite_PV_color)
  {
    last_pv = pv;
    if (hw_error)
      tft.setTextColor(ILI9341_RED);
    tft.fillRect(x_pv, y, tw, h, fillColor);
    tft.setCursor(x_pv, y);
    tft.printf("%05.2fC", pv);
    if (hw_error)
      tft.setTextColor(QATCH_BLUE_FG);
  }

  if (sp != last_sp)
  {
    last_sp = sp;
    tft.fillRect(x_sp, y, tw, h, fillColor);
    tft.setCursor(x_sp, y);
    tft.printf("%05.2fC", sp);
  }

  if (op != last_op)
  {
    last_op = op;
    tft.fillRect(x_op, y, tw - h, h, fillColor);
    tft.setCursor(x_op, y);
    tft.printf("%+04.0f", op);
  }

  //  tft.fillRect(0, rect_y + rect_h + 5 * pad, TFT_WIDTH, rect_h, fillColor);
  //  char buf[32]; // trailing NULL
  //  if (op == 0 || !l298nhb.active())
  //  {
  //    //    sprintf(buf, "PV:%sC SP:%2.2fC OP:%s", "--.--", sp, "[OFF]");
  //    sprintf(buf, "Temp Control is idle.");
  //  }
  //  else
  //  {
  //    sprintf(buf, "PV:%05.2fC SP:%05.2fC OP:%+04.0f", pv, sp, op);
  //  }
  //  uint16_t w = tft.measureTextWidth(buf);
  //  tft.setCursor((TFT_WIDTH - w) / 2, rect_y + rect_h + 5 * pad);
  //  tft.print(buf);

  if (op != 0)
  {
    //        Serial.print(op);
    //        Serial.print(", ");
    //        Serial.println(pct);
    if (pct != last_pct)
    {
      last_pct = pct;
      pct = abs(pct); // force to positive value
      // TODO on 6/27/23: Round edge of red/blue bar always to match grey area, widen gauge if need be for 1px per % range.
      if (op > 0)
      {
        // heating
        //        if (false) // (pct < rect_h / 2) // (pct < 100 - rect_r)
        //        {
        //          tft.fillRect(rect_x + 1, rect_y, pct + 1, rect_h, ILI9341_RED);
        //        }
        //        else
        //        {
        int r = rect_h / 2; // rect_r - (100 - pct); // min(pct-1, rect_r);
        int x = rect_x + 1;
        int y = rect_y;
        int w = pct + r + 1;
        int h = rect_h;
        tft.fillRect(x, y, w - r, h, ILI9341_RED);
        tft.fillCircleHelper(x + w - r - 1, y + r, r, 1, h - 2 * r - 1, ILI9341_RED);
        // fillRect(x+r, y, w-2*r, h, color); --> fillRect(x, y, w-r, h, color);
        // fillCircleHelper(x+w-r-1, y+r, r, 1, h-2*r-1, color);
        //         }
      }
      else
      {
        // cooling
        //        if (false) // (pct < rect_h / 2) // (pct < 100 - rect_r)
        //        {
        //          tft.fillRect(rect_x - (pct + 1), rect_y, pct + 1, rect_h, ILI9341_BLUE);
        //        }
        //        else
        //        {
        int r = rect_h / 2; // rect_r - (100 - pct); // min(pct-1, rect_r);
        int x = rect_x - (pct + r + 1) + 1;
        int y = rect_y;
        int w = pct + r + 1;
        int h = rect_h;
        tft.fillRect(x + r, y, w - r, h, ILI9341_BLUE);
        tft.fillCircleHelper(x + r, y + r, r, 2, h - 2 * r - 1, ILI9341_BLUE);
        // fillRect(x+r, y, w-r, h, color);
        // fillCircleHelper(x+r    , y+r, r, 2, h-2*r-1, color);
        //         }
      }
    }
  }

  //  uint16_t x, y, h; // , w;  // 'w' already declared in this context
  char line1[16]; // trailing NULL
  char line2[16]; // trailing NULL
  byte textSize = 28;
  uint16_t textColor = QATCH_BLUE_FG;

  byte label_state = l298nhb.getLabelState();
  //  Serial.print("label_state: ");
  //  Serial.println(label_state);
  if (label_state != last_temp_label || label_state == 0 || label_state == 2 || label_state == 3)
  {
    bool update_time_remaining_only = (last_temp_label == 2) && (label_state == 2);
    last_temp_label = label_state;
    line2[0] = '\0'; // set empty

    switch (label_state)
    {
    // Assign labels for each byte state
    case 0: // Temp Cycling
    {
      bool toggle_state = (getSystemTime() % 6000 <= 3000);
      if (toggle_state)
        sprintf(line1, "Temp Cycling");
      else if (l298nhb.getSignal())
        sprintf(line1, "COOLING");
      else
        sprintf(line1, "HEATING");
    }
    break;
    case 1: // Wait for Ready
      sprintf(line1, "Wait for Ready");
      break;
    case 2: // Ready in {X}
    {
      float time_remaining = l298nhb.getTimeRemaining();
      if (time_remaining < 0 || time_remaining > 60)
        time_remaining = 0; // wrapped
      if (time_remaining > 30)
        time_remaining = 30;
      sprintf(line1, "Ready in %2.0f", time_remaining);
    }
    break;
    case 3: // Ready
    {
      bool toggle_state = (getSystemTime() % 4000 <= 1000);
      if (toggle_state)
      {
        sprintf(line1, "Ready");
        textSize = 48;
      }
      else
      {
        sprintf(line1, "Press Start");
        sprintf(line2, "Then Apply Drop");
        textSize = 18;
      }
    }
    break;
    default:
      sprintf(line1, "Unknown State");
      textColor = ILI9341_RED;
      break;
    }

    //  if (hw_error)
    //  {
    //    sprintf(line1, "Temp Sensor Error");
    //    textColor = ILI9341_RED;
    //  }

    //  tft_wakeup();
    //  tft.fillScreen(ILI9341_BLACK);

    bool line_changed = (strcmp(line1, last_line_label) != 0);
    if (line_changed || label_state == 3)
    {
      strcpy(last_line_label, line1);

      // bool dp = (textSize == 28);
      if (textSize == 18)
        tft.setFont(Poppins_18_Bold);
      else if (textSize == 48)
        tft.setFont(Poppins_48_Bold);
      else
        tft.setFont(Poppins_28_Bold);

      if (label_state == 3)
      {
        bool toggle_state = (getSystemTime() % 4000 <= 1000);
        if (toggle_state)
        {
          textColor = QATCH_BLUE_FG; // Ready
        } else {
          toggle_state = (getSystemTime() % 1000 <= 500); // flash 'Apply Drop' message
          textColor = toggle_state ? ILI9341_GREEN : QATCH_BLUE_FG;
        }
      }
      tft.setTextColor(textColor);

      //  tft.setTextSize(textSize);

      //  x = (TFT_WIDTH - logo_image.width) / 2; // center
      //  y = (TFT_HEIGHT / 4) - (logo_image.height / 2); // middle-top
      //
      //  tft.writeRect(x, y, logo_image.width, logo_image.height, (uint16_t*)(logo_image.pixel_data));

      //  String line1 = tr ? "Ready" : "Temp Cycling...";
      //  char buff1[line1.length() + 1]; // trailing NULL
      //  line1.toCharArray(buff1, sizeof(buff1));

      w = tft.measureTextWidth(line1);
      h = tft.measureTextHeight(line1);
      x = (TFT_WIDTH - w) / 2;
      y = (3 * TFT_HEIGHT / 4) - h; // middle-bottom
      // if (!dp)
      //   y -= 10; // move "Ready" up a bit to keep it centered

      if (update_time_remaining_only)
      {
        pad = 3;
        x = 226; // += (20 * 9);
        w = 48;  // (20 * 2);
        //        Serial.println(x);
        //        Serial.println(y);
        //        Serial.println(w);
        //        Serial.println(h);
        //        Serial.println();

        tft.fillRect(x - pad, y - pad, w + pad, h + 2 * pad, fillColor);
        tft.setCursor(x, y);
        tft.print(line1[9]);  // "Ready In XX"
        tft.print(line1[10]); // "Ready In XX"
      }
      else
      {
        if (*line2) // is non-empty string?
        {
          y -= h + (textSize / 2); // move up to make room for second line
          h += h + (textSize / 2); // increase height to fill black rectangle for both lines
        }

        if (line_changed)
        {
          pad += 20; // matches 'dp' [larger font] - [smaller font]: 48 - 28 = 20 ( really 23, for safety pad)
          tft.fillRect(0, y - pad, TFT_WIDTH, h + 2 * pad, fillColor == ILI9341_BLACK ? ILI9341_BLACK : ILI9341_DARKGREY);
        }

        tft.setCursor(x, y);
        tft.print(line1);

        if (*line2) // is non-empty string?
        {
          // Serial.print("Printing: ");
          // Serial.println(line2);
          w = tft.measureTextWidth(line2);
          h = tft.measureTextHeight(line2);
          x = (TFT_WIDTH - w) / 2;
          y += h + (textSize / 2); // middle-bottom
          tft.setCursor(x, y);
          tft.print(line2);
        }
      }
    }
  }
  //    }
  //  }
}

void tft_initialize()
{
  if (HW_REV_MATCH(HW_REVISION_0) || PID_IS_SECONDARY(NVMEM.pid))
  {
    // NOTE: a small delay is still required here to
    //       allow the DAC to settle after waking up
    delayMicroseconds(750);
    return;
  }

  //  tft_initialize_styleB();
  //}

  // void tft_initialize_styleA()
  //{
  //  tft_tempbusy(); // calls wakeup()
  //   tft.fillRect(0, 0, TFT_WIDTH, 20, ILI9341_WHITE);
  //   tft.drawFastHLine(0, 21, TFT_WIDTH, ILI9341_BLACK);
  //   tft.setCursor(3, 3);
  //   tft.setTextSize(2);
  //   tft.setTextColor(ILI9341_BLACK);
  //   tft.print("Initializing...");
  // }

  // void tft_initialize_styleB()
  //{

  //unsigned long start = micros();
  uint16_t x, y, w, h; // , pad;

  tft_wakeup();
  tft.fillScreen(ILI9341_BLACK);
  tft.setFont(Poppins_32_Bold);
  tft.setTextColor(QATCH_BLUE_FG);
  //  tft.setTextSize(3);

  //  x = (TFT_WIDTH - initialize_icon.width) / 2; // center
  //  y = (TFT_HEIGHT / 4) - (initialize_icon.height / 2); // middle-top
  //
  //  tft.writeRect(x, y, initialize_icon.width, initialize_icon.height, (uint16_t*)(initialize_icon.pixel_data));

  String line1 = "Initializing";
  char buff1[line1.length() + 1]; // trailing NULL
  line1.toCharArray(buff1, sizeof(buff1));

  w = tft.measureTextWidth(buff1); // TODO: hard-code, should be a fixed number
  h = tft.measureTextHeight(buff1);
  x = (TFT_WIDTH - w) / 2;
  y = (3 * TFT_HEIGHT / 4) - h; // middle-bottom

  tft.setCursor(x, y);
  tft.print(line1);

  //  pad = 10;
  //  x = (TFT_WIDTH - initialize_icon.width) / 2; // center
  //  y = TFT_HEIGHT - (y + h) - pad; // 'y' and 'h' refer to the text placement, not the image

  x = ICON_X;
  y = ICON_Y;
  tft.writeRect(x, y, initialize_icon.width, initialize_icon.height, (uint16_t *)(initialize_icon.pixel_data));

  // unsigned long stop = micros();
  // client->print("TFT Wakeup Duration: ");
  // client->print(stop - start);
  // client->println("us");
}

void tft_measure()
{
  if (HW_REV_MATCH(HW_REVISION_0) || PID_IS_SECONDARY(NVMEM.pid))
    return;

  //  tft_measure_styleB();
  //}

  // void tft_measure_styleA()
  //{
  //   tft_wakeup();
  //   tft.fillRect(0, 0, TFT_WIDTH, 20, ILI9341_WHITE);
  //   tft.drawFastHLine(0, 21, TFT_WIDTH, ILI9341_BLACK);
  //   tft.setCursor(3, 3);
  //   tft.setTextSize(2);
  //   tft.setTextColor(ILI9341_BLACK);
  //   tft.print("Measuring...");
  //  Serial.println("Measuring...");
  // }

  // void tft_measure_styleB()
  //{
  uint16_t x, y, w, h; // , pad;

  tft_wakeup();
  tft.fillScreen(ILI9341_BLACK);
  tft.setFont(Poppins_32_Bold);
  tft.setTextColor(QATCH_BLUE_FG);
  //  tft.setTextSize(3);

  //  x = (TFT_WIDTH - measure_icon.width) / 2; // center
  //  y = (TFT_HEIGHT / 4) - (measure_icon.height / 2); // middle-top
  //
  //  tft.writeRect(x, y, measure_icon.width, measure_icon.height, (uint16_t*)(measure_icon.pixel_data));

  String line1 = "Measuring";
  char buff1[line1.length() + 1]; // trailing NULL
  line1.toCharArray(buff1, sizeof(buff1));

  w = tft.measureTextWidth(buff1); // TODO: hard-code, should be a fixed number
  h = tft.measureTextHeight(buff1);
  x = (TFT_WIDTH - w) / 2;
  y = (3 * TFT_HEIGHT / 4) - h; // middle-bottom

  tft.setCursor(x, y);
  tft.print(line1);

  //  pad = 10;
  //  x = (TFT_WIDTH - measure_icon.width) / 2; // center
  //  y = TFT_HEIGHT - (y + h) - pad; // 'y' and 'h' refer to the text placement, not the image

  x = ICON_X;
  y = ICON_Y;
  tft.writeRect(x, y, measure_icon.width, measure_icon.height, (uint16_t *)(measure_icon.pixel_data));
}

#else // NOT USE_ILI9341

// create dummy public function stubs
void tft_wakeup() {}
void tft_screensaver() {}
void tft_splash(bool dp) {}
void tft_identify(bool identifying) {}
void tft_idle() {}
// void tft_testmode() { }
// void tft_tempbusy() { }
void tft_cooldown_start() {}
void tft_cooldown() {}
void tft_tempcontrol() {}
void tft_initialize() {}
void tft_measure() {}

#endif

/************************** TEESNY36 ***************************/

#if HW_MATCH(TEENSY36)

unsigned long getSystemTime()
{
  return millis();
}

#endif

/************************** TEENSY41 ***************************/

#if HW_MATCH(TEENSY41)

bool search_for_dhcp()
{
  unsigned long startMillis = millis();
  unsigned long timeout = 1000;
  IPAddress ip = Ethernet.localIP(); // restored if no offer
  last_dhcp_search = startMillis;

  fnet_netif_desc_t netif = fnet_netif_get_default();

  if (dhcp_enabled)
  {
    fnet_dhcp_cln_release(fnet_dhcp_cln_get_by_netif(netif));
  }

  if (!fnet_dhcp_cln_is_enabled(fnet_dhcp_cln_get_by_netif(netif)))
  {
    static fnet_dhcp_cln_params_t dhcp_params; // DHCP intialization parameters
    dhcp_params.netif = netif;
    // Enable DHCP client.
    if (fnet_dhcp_cln_init(&dhcp_params))
    {
      fnet_dhcp_cln_set_response_timeout(fnet_dhcp_cln_get_by_netif(netif), 2500);
      // Register DHCP event handler callbacks.
      //          fnet_dhcp_cln_set_callback_updated(fnet_dhcp_cln_get_by_netif(netif), dhcp_cln_callback_updated, NULL);
      //          fnet_dhcp_cln_set_callback_discover(fnet_dhcp_cln_get_by_netif(netif), dhcp_cln_callback_updated, NULL);
      // Serial.println("DHCP initialization done!");
    }
    else
    {
      // Serial.println("ERROR: DHCP initialization failed!");
    }
  }

  while (!fnet_dhcp_cln_is_enabled(fnet_dhcp_cln_get_by_netif(netif)))
  {
    // Wait for dhcp initialization
    if (millis() >= startMillis + timeout)
      break;
  }
  while (Ethernet.localIP() == IPAddress(0, 0, 0, 0))
  {
    // Wait for IP address
    if (millis() >= startMillis + timeout)
      break;
  }
  if (millis() >= startMillis + timeout) // timeout, no offer made
  {
    if (ip == IPAddress(0, 0, 0, 0))
      ip = IPAddress(169, 254, 73, mac[5]);
    Ethernet.begin(mac, ip); // also releases dhcp cln
    Ethernet.setSubnetMask(SUBNET_MASK);
    // Serial.print("Ethernet configured with static IP ");
    // Serial.println(Ethernet.localIP());
    return false;
  }

  // Serial.print("Ethernet configured with assigned IP ");
  // Serial.println(Ethernet.localIP());
  return true;
}

bool connect_to_ethernet()
{
  // initial assumptions
  net_error = false;
  dhcp_enabled = false;

  // RST_N for PHY chip is on B0_14 (GPIO2.IO[14]) per Teensy 4.1 schematics
  // Per DP83825I datasheet: The RST_N pin is an input with internal pull-up
  // NOTE: On Teensy 4.0-4.1, GPIO7 is used for non-DMA access to GPIO2 pins
  bool phy_chip_exists = GPIO7_PSR & (1 << 14); // true if PHY chip present
  if (!phy_chip_exists) 
  {
    client->println("HW Variant: TEENSY41_NE (Without Ethernet Chip)");
    delay(3000); // pause to show boot screen
    net_error = true;
    return false;
  }
  else
  {
    client->println("HW Variant: TEENSY41 (With Ethernet Chip)");
    // continue;
  }

  // Check 'Ethernet_EN' feature bit in NVMEM
  if (NVMEM.Ethernet_EN == 0) // not enabled (aka: disabled)
  {
    client->println("Ethernet_EN is not set in NVMEM. Not initializing Ethernet chip.");
    delay(3000); // pause to show boot screen
    net_error = true;
    return false;
  }

  // start the Ethernet connection and the server:
  Ethernet.setStackHeap(1024 * 128);     // Set stack size to 128k
  Ethernet.setSocketSize(1024 * 4);      // Set buffer size to 4k
  Ethernet.setSocketNum(MAX_IP_CLIENTS); // Set number of allowed sockets

  // static int begin(uint8_t *mac, unsigned long timeout = 60000, unsigned long responseTimeout = 4000);
  int e_result = Ethernet.begin(mac, 6000, 2500);

  // This allows forwarded NTP packets to be heard
  Ethernet.setSubnetMask(SUBNET_MASK);

  // Check for Ethernet hardware present
  if (Ethernet.hardwareStatus() == EthernetNoHardware)
  {
    // Serial.println("Ethernet hardware was not found.");
    net_error = true;
  }
  // Check for Ethernet cable present
  if (Ethernet.linkStatus() == LinkOFF)
  {
    // Serial.println("Ethernet cable is not connected.");
    net_error = true;
  }
  // Configure for static IP address
  if (net_error)
  {
    // do nothing
  }
  else if (e_result == 0)
  {
    IPAddress ip = IPAddress(169, 254, 73, mac[5]);
    Ethernet.begin(mac, ip);
    Ethernet.setSubnetMask(SUBNET_MASK);
    // Serial.print("Ethernet configured with static IP ");
    // Serial.println(Ethernet.localIP());
  }
  else // DHCP connect succeeded
  {
    dhcp_enabled = search_for_dhcp(); // set dhcp server param
    // Serial.print("Ethernet configured with assigned IP ");
    // Serial.println(Ethernet.localIP());
  }

  // start the servers
  Udp.begin(NTP_PORT);
  server.begin();

  return (Ethernet.linkStatus() == LinkON); // is_connected
}

unsigned long getSystemTime()
{
  return getSystemTime(false);
}

unsigned long getSystemTime(bool print_status)
{
  // Return corrected milliseconds using NTP time sync
  // TODO use elapsedMillis to handle rollover?
  // https://github.com/pfeerick/elapsedMillis/wiki

  // this will overflow, but the relative deltas work out
  unsigned long elapsed = millis() - last_TOI_TS;
  long drift_correction = 0;

  long period = TS_SYNC_PERIOD / drift_TS;
  // avoid doing negative division here
  drift_correction = elapsed / abs(period);
  if (period < 0)
    drift_correction *= -1;

  if (print_status)
  {
    client->printf("NOW DRIFT:  %i\n", drift_correction);
  }

  return 1000 * last_EPOCH_sec + last_EPOCH_ms + elapsed + drift_correction;
}

bool isTimeForSync()
{
  // Is it time for a new sync from the time server?
  // Condition #1: just booted, never been sync'd
  // Condition #2: millis() counter wrapped to zero
  // Condition #3: normal time sync expiration renew
  if (last_RX_TS == 0)
    return true;
  if (millis() < last_RX_TS)
    return true;
  if (millis() > last_RX_TS + TS_SYNC_PERIOD)
  {
    if (NTP_local_master)
      return true;
    else
    {
      // slaves wait for master to timeout before trying (with dither)
      return (millis() > last_RX_TS + TS_SYNC_PERIOD +
                             (TS_RETRY_COUNT * TS_RETRY_PERIOD) +
                             Ethernet.localIP()[3]);
    }
  }

  return false;
}

int checkForTimeSyncUpdates()
{
  int result = 0; // nothing changed, tx only, rx only, both tx/rx

  if (isTimeForSync())
  {
    if (!NTP_pending) // No packet in-flight
    {
      if (NTP_retries < TS_RETRY_COUNT) // No timeout
      {
        // send an NTP packet to a time server
        sendNTPpacket(dhcp_enabled ? timeServer : apipaServer);
        last_TX_TS = millis();
        NTP_pending = true;
        result = 1;
      }
      else // NTP retries exceeded (wait until next sync period)
      {
        last_RX_TS = millis(); // set to prevent more retries
        NTP_retries = 0;
        result = 4;

        // did the network change without us knowing?
        dhcp_enabled = search_for_dhcp();
      }
    }
    else // Packet already in-flight
    {
      // Check for roundtrip timeout
      if (millis() - last_TX_TS > TS_RETRY_PERIOD)
      {
        NTP_pending = false;
        NTP_retries++;
        result = 3;
      }
    }
  }

  if (Udp.parsePacket())
  {
    // We've received a packet, read the data from it
    Udp.read(packetBuffer, NTP_PACKET_SIZE); // read the packet into the buffer

    bool this_is_an_ntp_cmd = ((packetBuffer[0] == 0xE3) && (packetBuffer[1] == 0x00) &&
                               (packetBuffer[2] == 0x06) && (packetBuffer[3] == 0xEC));
    bool this_is_a_ping_cmd = ((packetBuffer[0] == 'P') && (packetBuffer[1] == 'I') &&
                               (packetBuffer[2] == 'N') && (packetBuffer[3] == 'G'));

    IPAddress remote_ip = Udp.remoteIP();
    IPAddress local_ip = Ethernet.localIP();
    IPAddress subnet = Ethernet.subnetMask();
    IPAddress broadcast_ip = IPAddress((local_ip[0] | ~subnet[0]),
                                       (local_ip[1] | ~subnet[1]),
                                       (local_ip[2] | ~subnet[2]),
                                       (local_ip[3] | ~subnet[3]));

    // Check for incoming UDP commands that indicate we are on APIPA network configuration
    if (dhcp_enabled && remote_ip[0] == 169 && remote_ip[1] == 254)
    { // remote device is APIPA; but we are not APIPA
      // we think we have DHCP, but somehow the network lost it;
      // other devices know, so we should fallback to APIPA config
      bool dhcp_was = dhcp_enabled;
      dhcp_enabled = search_for_dhcp();

      // network changed: update local network params before processing packet
      if (dhcp_was != dhcp_enabled)
      {
        local_ip = Ethernet.localIP();
        subnet = Ethernet.subnetMask();
        broadcast_ip = IPAddress((local_ip[0] | ~subnet[0]),
                                 (local_ip[1] | ~subnet[1]),
                                 (local_ip[2] | ~subnet[2]),
                                 (local_ip[3] | ~subnet[3]));
      }
    }

    // Throw away this packet if it is a local outgoing NTP request
    if (this_is_an_ntp_cmd)
      return result; // toss packet

    // Check for NTP packets from remote server (not local forwarding)
    if (((remote_ip[0] & subnet[0]) != (local_ip[0] & subnet[0])) ||
        ((remote_ip[1] & subnet[1]) != (local_ip[1] & subnet[1])) ||
        ((remote_ip[2] & subnet[2]) != (local_ip[2] & subnet[2])) ||
        ((remote_ip[3] & subnet[3]) != (local_ip[3] & subnet[3])))
    {
      unsigned long fwd_start = millis();
      int count = 0;
      while (millis() < fwd_start + 5 && count < 10)
      {
        delayMicroseconds(250);

        // this UDP packet came directly from the time server (and not a local source)
        // so forward it on to the other local devices too
        Udp.beginPacket(broadcast_ip, NTP_PORT);
        Udp.write(packetBuffer, NTP_PACKET_SIZE);
        Udp.endPacket();
        count++;
      }
      // Serial.print("Sent forward packet count: ");
      // Serial.println(count);

      // Increment stats
      NTP_remote_pkts++;
      NTP_local_master = true;
    }
    else
    {
      // this UDP packet came from the local network (possibly this same device)
      // so do not process or propogate your own forwarded packets!
      if (remote_ip[3] == local_ip[3])
        return result; // toss packet

      // also only process one incoming NTP packet for every 100 ms
      if (millis() - last_RX_TS < 100)
        return result; // toss packet

      // also special case to handle PING requests from other devices
      if (this_is_a_ping_cmd)
      {
        // Serial.print("PING request from: ");
        // Serial.println(remote_ip);
        delayMicroseconds(250 + (2550 * local_ip[2]) + (10 * local_ip[3]));
        Udp.beginPacket(remote_ip, NTP_PORT);
        Udp.write(packetBuffer, NTP_PACKET_SIZE);
        Udp.endPacket();
        return result;
      }

      // Increment stats
      NTP_local_pkts++;
      NTP_local_master = false;
    }

    // clear NTP counters
    NTP_pending = false;
    NTP_retries = 0;
    result = 2;

    // the timestamp starts at byte 40 of the received packet and is eight bytes,
    // or four words, long. First, extract the four words:
    unsigned long tsHighWord = word(packetBuffer[40], packetBuffer[41]);
    unsigned long tsLowWord = word(packetBuffer[42], packetBuffer[43]);
    unsigned long msHighWord = word(packetBuffer[44], packetBuffer[45]);
    unsigned long msLowWord = word(packetBuffer[46], packetBuffer[47]);

    // combine the first four bytes (two words) into a long integer
    // this is NTP time (seconds since Jan 1 1900):
    unsigned long secsSince1900 = tsHighWord << 16 | tsLowWord;
    // now convert NTP time into everyday time:
    // Unix time starts on Jan 1 1970. In seconds, that's 2208988800:
    const unsigned long seventyYears = 2208988800UL;
    // subtract seventy years:
    unsigned long epoch = secsSince1900 - seventyYears;

    // combine the last four bytes (two words) into a long integer
    // this is NTP time sub-second resolution (fractional seconds):
    unsigned long fracsecs = msHighWord << 16 | msLowWord;
    // convert fractional seconds to standardized milliseconds:
    unsigned short mssecs = ((fracsecs >> 7) * 125 + (1UL << 24)) >> 22;

    if (getSyncState() == 1) // only if sync is needed
    {
      // this will overflow, but the relative deltas work out
      drift_TS = (((1000 * epoch) + mssecs) - millis()) -
                 (((1000 * last_EPOCH_sec) + last_EPOCH_ms) - last_TOI_TS);
    }

    // Store results
    last_RX_TS = last_TOI_TS = millis();
    last_EPOCH_sec = epoch;
    last_EPOCH_ms = mssecs;

    // DEBUG PRINT STATEMENTS (COMMENT OUT)
    // Serial.print("Seconds since Jan 1 1900 = ");
    // Serial.println(secsSince1900);
    // Serial.print("Unix time = ");
    // Serial.println(epoch);
    // Serial.print("Frac secs = " );
    // Serial.println(mssecs);
    // Serial.print("Millis = " );
    // Serial.println(millis());
    // Serial.print("drift (ms) = ");
    // Serial.println(drift_TS);
    // Serial.print("Offset millis = ");
    // Serial.println(t_offset);
    // Serial.print("Corrected millis = ");
    // Serial.println(millis() - t_offset);
    // Serial.print("Sync millis = ");
    // Serial.println(getSystemTime());
    // getSystemTime(true);
  }

  return result;
}

// send an NTP request to the time server at the given address
void sendNTPpacket(const char *address)
{
  // set all bytes in the buffer to 0
  memset(packetBuffer, 0, NTP_PACKET_SIZE);
  // Initialize values needed to form NTP request
  // (see URL above for details on the packets)
  packetBuffer[0] = 0b11100011; // LI, Version, Mode
  packetBuffer[1] = 0;          // Stratum, or type of clock
  packetBuffer[2] = 6;          // Polling Interval
  packetBuffer[3] = 0xEC;       // Peer Clock Precision
  // 8 bytes of zero for Root Delay & Root Dispersion
  packetBuffer[12] = 49;
  packetBuffer[13] = 0x4E;
  packetBuffer[14] = 49;
  packetBuffer[15] = 52;

  // all NTP fields have been given values, now
  // you can send a packet requesting a timestamp:
  Udp.beginPacket(address, 123); // NTP requests are to port 123
  Udp.write(packetBuffer, NTP_PACKET_SIZE);
  Udp.endPacket();
}

unsigned long getNowEpoch()
{
  unsigned long elapsed_sec = (millis() - last_TOI_TS) / 1000;
  return (last_EPOCH_sec + elapsed_sec);
}

short getSyncState()
{
  /* getSyncState() return code:
     -1   no-sync
      0   in-sync
      1   out-of-sync (once)
      2   out-of-sync (two+)
  */
  if (last_EPOCH_sec == 0)
    return -1;
  long secsSinceLastSync = getNowEpoch() - last_EPOCH_sec;
  if ((TS_SYNC_PERIOD / 1000) > secsSinceLastSync)
    return 0;
  if ((TS_SYNC_PERIOD / 1000) * 2 > secsSinceLastSync)
    return 1;
  return 2;
}

#endif
