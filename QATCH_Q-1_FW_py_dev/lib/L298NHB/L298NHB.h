// this library is public domain. enjoy!
// https://learn.adafruit.com/thermocouple/

#ifndef ANALOG_L298NHB_H
#define ANALOG_L298NHB_H

#include "Arduino.h"

#define MAX_PWR_HEAT 255
#define MAX_PWR_COOL 255

/**************************************************************************/
/*!
    @brief  Class for communicating with thermocouple sensor
*/
/**************************************************************************/
class L298NHB
{

public:
  L298NHB(int8_t M1, int8_t E1, int8_t M2, int8_t E2, bool initM, byte initE);
  void setTuning(float cp, float ci, float cd, float hp, float hi, float hd);
  float getTuning(byte i);
  bool setSignal(bool v);
  bool getSignal(void);
  bool targetReached();
  float setTarget(float v);
  float getTarget(void);
  float setAmbient(float v);
  float getAmbient(void);
  byte setPower(byte p);
  byte getPower(void);
  float timeStable(void);
  float timeElapsed(void);
  float getMinTemp(void);
  float getMaxTemp(void);
  int8_t update(float t);
  void shutdown(void);
  void wakeup(void);
  bool active(void);
  void resetMinMax(void);
  byte getLabelState(void);
  float getTimeRemaining(void);

private:
  int8_t m0, e0, m1, e1, m2, e2;
  bool signal;   // heating or cooling
  byte power;    // peltier power (pwm)
  float target;  // temperature (celcius)
  float ambient; // temperature (celcius)
  bool invert;   // for inverse modes (when ambient lies)
  byte is_stable;
  unsigned long t_start, t_last;
  unsigned long time_of_last_instability;
  byte label_state;
  float t_stable; // time to stable temp
  float temp_min, temp_max;
  float PID_error, PID_error_avg, previous_error;
  uint8_t PID_error_count;
  bool pid_retuned = false;
  
  // PID constants
  float kp = 10;
  float ki = 0.1;
  float kd = 100;

  float kpc = kp * MAX_PWR_COOL / 255;
  float kic = ki * MAX_PWR_COOL / 255;
  float kdc = kd * MAX_PWR_COOL / 255;

  float kph = kp * MAX_PWR_HEAT / 255;
  float kih = ki * MAX_PWR_HEAT / 255;
  float kdh = kd * MAX_PWR_HEAT / 255;

  float PID_p = 0;
  float PID_i = 0;
  float PID_d = 0;
};

#endif
