// Written for use with QATCH devices by Alex Ross on 9/21/2021

#include "L298NHB.h"

#define MAX_TEMP_DELTA 1.0
#define D_REFRESH_RATE 10 // secs

/**************************************************************************/
/*!
    @brief  Initialize a LTC1658 sensor
    @param   M1 The Arduino pin connected to Motor Signal Pin 1
    @param   E1 The Arduino pin connected to Motor Enable Pin 1
    @param   M2 The Arduino pin connected to Motor Signal Pin 2
    @param   E2 The Arduino pin connected to Motor Enable Pin 2
    @param   initM The starting state of the Motor Signal Pins
    @param   initE The starting state of the Motor Enable Pins
*/
/**************************************************************************/
L298NHB::L298NHB(int8_t M1, int8_t E1, int8_t M2, int8_t E2, bool initM, byte initE)
{
  m0 = initM;
  e0 = initE;
  m1 = M1;
  e1 = E1;
  m2 = M2;
  e2 = E2;
  signal = m0;
  power = e0;
  target = 25.0;
  ambient = 25.0;
  invert = false;

  if (initM)
  {
    // start in COOL mode, prep PID constants
    kp = kpc;
    ki = kic;
    kd = kdc;
  }
  else
  {
    // start in HEAT mode, prep PID constants
    kp = kph;
    ki = kih;
    kd = kdh;
  }

  // define pin modes
  pinMode(m1, OUTPUT);
  pinMode(e1, OUTPUT);
  pinMode(m2, OUTPUT);
  pinMode(e2, OUTPUT);

  // init signal
  digitalWrite(m1, signal);
  digitalWrite(m2, signal);

  delayMicroseconds(3); // wait for settle

  // init power
  analogWrite(e1, power);
  analogWrite(e2, power);

  shutdown();
}

/**************************************************************************/
/*!
    @brief  Set the temperature PID tuning parameters
    @param   cp proportional gain constant (cooling)
    @param   ci integral gain constant (cooling)
    @param   cd derivative gain constant (cooling)
    @param   hp proportional gain constant (heating)
    @param   hi integral gain constant (heating)
    @param   hd derivative gain constant (heating)
    @returns void
*/
/**************************************************************************/
void L298NHB::setTuning(float cp, float ci, float cd, float hp, float hi, float hd)
{
  pid_retuned = true;
  // cool PID:
  kpc = cp;
  kic = ci;
  kdc = cd;
  // heat PID:
  kph = hp;
  kih = hi;
  kdh = hd;
}

/**************************************************************************/
/*!
    @brief  Get the temperature PID tuning parameters
    @param   i index of PID tuning parameter
    @returns float PID param (of given index)
*/
/**************************************************************************/
float L298NHB::getTuning(byte i)
{
  switch (i)
  {
  case 0:
    return kpc;
  case 1:
    return kic;
  case 2:
    return kdc;
  case 3:
    return kph;
  case 4:
    return kih;
  case 5:
    return kdh;
  default:
    return 0;
  }
}

/**************************************************************************/
/*!
    @brief  Set the temperature signal
    @param   v signal (heating or cooling)
    @returns signal (heating or cooling)
*/
/**************************************************************************/
bool L298NHB::setSignal(bool v)
{
  bool s = digitalRead(m1);
  signal = v;

  if (s != signal)
  {
    // turn OFF before signal switch to protect transistors!
    byte restore = getPower();
    if (restore != 0)
      setPower(0);
    // else, power already set to zero (no need to turn off)

    delayMicroseconds(3); // wait for settle
    digitalWrite(m1, signal);
    digitalWrite(m2, signal);
    delayMicroseconds(3); // wait for settle

    // restore power after signal switch is completed
    if (restore != 0)
      setPower(restore);
    // else, power already set to zero (no need to restore)
  }

  s = digitalRead(m2);
  return s;
}

/**************************************************************************/
/*!
    @brief  Get the temperature signal
    @returns signal (heating or cooling)
*/
/**************************************************************************/
bool L298NHB::getSignal(void)
{
  return signal;
}

/**************************************************************************/
/*!
    @brief  Set the temperature target
    @param   t target (0.0 - 60.0, 0.25 resolution)
    @returns target (0.0 - 60.0, 0.25 resolution)
*/
/**************************************************************************/
float L298NHB::setTarget(float t)
{
  is_stable = false;
  t_start = millis();
  resetDeltaTime();
  ki = (signal ? kic : kih);
  PID_i = 0;
  PID_d = 0;
  t_stable = 0;
  resetMinMax();
  time_of_last_instability = millis(); // not zero, but time of TEC start
  label_state = 0;                     // STATE 0: Print "Temp Cycling"
  invert = false;
  target = round(t / 0.25) * 0.25;
  target = max(0, min(60, target));
  PID_error_avg = 0;
  PID_error_count = 0;
  previous_error = 0;
  return target;
}

/**************************************************************************/
/*!
    @brief  Get the temperature target
    @returns target (0.0 - 60.0, 0.25 resolution)
*/
/**************************************************************************/
float L298NHB::getTarget(void)
{
  return target;
}

/**************************************************************************/
/*!
    @brief  Set the temperature ambient
    @param   t target (0.0 - 60.0, 0.25 resolution)
    @returns target (0.0 - 60.0, 0.25 resolution)
*/
/**************************************************************************/
float L298NHB::setAmbient(float t)
{
  ambient = round(t / 0.25) * 0.25;
  ambient = max(0, min(60, ambient));
  return ambient;
}

/**************************************************************************/
/*!
    @brief  Get the temperature target
    @returns target (0.0 - 60.0, 0.25 resolution)
*/
/**************************************************************************/
float L298NHB::getAmbient(void)
{
  return ambient;
}

/**************************************************************************/
/*!
    @brief  Set the Peltier power
    @param   p power (0 - 255)
    @returns power (0 - 255)
*/
/**************************************************************************/
byte L298NHB::setPower(byte p)
{
  bool s = analogRead(e1);
  power = p;

  if (s != power)
  {
    analogWrite(e1, power);
    analogWrite(e2, power);
  }

  s = analogRead(e2);
  return s;
}

/**************************************************************************/
/*!
    @brief  Get the Peltier power
    @returns power (0 - 255)
*/
/**************************************************************************/
byte L298NHB::getPower(void)
{
  return power;
}

bool L298NHB::targetReached(void)
{
  return is_stable;
}

float L298NHB::timeStable(void)
{
  return t_stable;
}

float L298NHB::timeElapsed(void)
{
  if (!active())
    return 0;
  return (millis() - t_start) / 1000.0;
}

float L298NHB::getMinTemp(void)
{
  return temp_min;
}

float L298NHB::getMaxTemp(void)
{
  return temp_max;
}

void L298NHB::resetMinMax(void)
{
  temp_min = 60;
  temp_max = 0;
}

byte L298NHB::getLabelState(void)
{
  return label_state;
}

float L298NHB::getTimeRemaining(void)
{
  unsigned long ready_at = time_of_last_instability + 60 * 1000;
  unsigned long time_remain = ready_at - millis();
  return (float)(time_remain / 1000.0) + 1.0; // convert to seconds, round up
}

void L298NHB::resetDeltaTime(void)
{
  t_last = millis();
}

/**************************************************************************/
/*!
    @brief  Task to maintain a given temperature target
    @returns mode (-1 = cooling, 0 = idle, 1 = heating)
*/
/**************************************************************************/
int8_t L298NHB::update(float t)
{
  int8_t mode = 0; // idle
  float t_diff = t - getTarget();
  uint8_t max_pwr = 0;

  if (is_stable)
  {
    if (t > temp_max)
      temp_max = t;
    if (t < temp_min)
      temp_min = t;
  }

  unsigned long t_now = millis();
  float elapsedTime = (t_now - t_start) / 1000.0;

  // when a new target is provided by the user:
  // check that we're trending in the right direction
  // and if not, change modes to trend temp correctly
  // if (power == 1 || abs(t_diff) > 5) // if min pwr doesn't cut it or temp >5C away from target
  // {
  //   invert = false; // assume no invert, unless needed per below
  //   if (abs(target - ambient) > 5) // target must be at least 5C away from ambient to invert;
  //   { // otherwise, always pick signal based on difference to target
  //     if (t_diff > 0) // temp too high
  //     {
  //       if (target > ambient) // but heating
  //       {
  //         invert = true; // switch modes (heat to cool)
  //       }
  //     }
  //     else // temp too low
  //     {
  //       if (target <= ambient) // but cooling
  //       {
  //         invert = true; // switch modes (cool to heat)
  //       }
  //     }
  //   }
  // }
  if ((!is_stable) &&                      // we were still trending towards target
      (abs(t_diff) <= MAX_TEMP_DELTA / 2)) // and are now very close to target
  {
    t_stable = elapsedTime;
    is_stable = true; // target reached
    // PID_i /= 2; // cut in half, grow back what's needed
  }
  // end is_stable code

  // treat ambient as actual temp when target is within 5C of ambient
  float reference = ambient;
  // never do this with PID tuning control
  // if (true || abs(target - ambient) < 5 || abs(target - t) < 5) // && (invert || power == 1)) // && elapsedTime <= 1)
  // {
  //   // invert = false;
  //   // if ((target < ambient && mode == 1) ||
  //   //     (target > ambient && mode == -1))
  //   //   invert = true;
  //   mode = getSignal() ? -1 : 1;
  //   reference = t - (mode * MAX_TEMP_DELTA); // this may trigger mode switch if operating near ambient
  // }

  if (t_diff != 0)
  {
    if ((!invert && (target > reference)) || (invert && (target < reference)))
    {
      // start from low power when changing from cool to heat
      if (0 != getSignal() || pid_retuned)
      {
        setPower(0); // briefly, always set to non-zero again later
        pid_retuned = false;
        kp = kph;
        ki = kih;
        kd = kdh;
        PID_i = 0;
        PID_d = 0;
      }
      // switch to heat
      setSignal(0);
      // mode is heat
      mode = 1;
      max_pwr = MAX_PWR_HEAT;
      // positive error means more power
      PID_error = -t_diff; // target - t
      // Calculate the P value
      PID_p = kp * (target - t); // compare SP vs PV (not ambient), w/o abs()
    }
    else
    {
      // start from low power when changing from heat to cool
      if (1 != getSignal() || pid_retuned)
      {
        setPower(0); // briefly, always set to non-zero again later
        pid_retuned = false;
        kp = kpc;
        ki = kic;
        kd = kdc;
        PID_i = 0;
        PID_d = 0;
      }
      // switch to cool
      setSignal(1);
      // mode is cool
      mode = -1;
      max_pwr = MAX_PWR_COOL;
      // positive error means more power
      PID_error = t_diff; // t - target
      // Calculate the P value
      PID_p = kp * (t - target); // compare PV (not ambient) vs SP, w/o abs()
    }

    // Calculate the I value in a range on +-5
    if (true || abs(PID_error) <= 5)
    {
      // speed up when diff is outside -0.50C to +0.50C range
      // byte factor = (abs(t_diff) > 0.50 ? 3 : 1); // 3x faster when drifting >0.5C away from target
      float factor = min(abs(t_diff), 1); // i-factor is proportional to difference error with max = 1
                                          // so, when PV is close to SP (within 1C) i-factor decreases
      if (((mode == -1) && (t_diff < -0.25)) ||
          ((mode == +1) && (t_diff > +0.25)))
        factor *= 3; // 3x faster when overshoot
      factor = 1;    // always 1, fixed (override above logic)
      ki = factor * (signal ? kic : kih);
      PID_i = PID_i + (ki * PID_error);
      // PID_i = max(-max_pwr + (PID_p + PID_d), min(max_pwr - (PID_p + PID_d), PID_i)); // prevent runaway condition
      // Serial.print("\nPID_i:");
      // Serial.println(PID_i);
    }

    // Calculate the D value
    PID_error_count++;
    PID_error_avg += PID_error;
    float deltaTime = (t_now - t_last) / 1000.0;
    if (deltaTime > D_REFRESH_RATE)
    {
      PID_error_avg /= PID_error_count;
      // Serial.print("PID_error_avg: "); // DEBUG
      // Serial.println(PID_error_avg);   // DEBUG
      if (previous_error != 0)
        PID_d = kd * ((PID_error_avg - previous_error) / deltaTime);
      previous_error = PID_error_avg;
      PID_error_count = 0;
      PID_error_avg = 0;
      t_last = t_now;
      if (false) // disable prior logic: (abs(t_diff) < 2 * MAX_TEMP_DELTA || is_stable)
      {
        // Once within 2x stable range, use PID_d to aggresively prevent overshoot
        // PID_i will grow back what is required to keep the setpoint stablized...
        PID_d = PID_d + map(PID_error, -MAX_TEMP_DELTA, 2 * MAX_TEMP_DELTA, min(-1, (int)(-PID_i)), 0);
      }
    }

    // Final total PID value is the sum of P + I + D
    int PID_value = (int)(PID_p + PID_i + PID_d);

    // if (PID_error >= 5)
    // {
    //   PID_value = max_pwr;
    // }
    // if (PID_value <= 1)
    // {
    //   PID_value += power;
    // }

    // switch direction if PID is wanting to drive the other direction
    if (PID_value <= 0)
      invert = !invert;

    if (((target < reference && mode == 1) ||
         (target > reference && mode == -1)) &&
        !invert && false) // disable prior logic with programmable PID tuning
    {
      // if we should be cooling, but we are actually heating -or-
      // if we should be heating, but we are actually cooling -so-
      // then, go very slowly (do not rush it) - ignore if inverse
      PID_value = 1;
    }
    power = max(1, min(max_pwr, PID_value)); // prevent rollover
    setPower(power);
  }
  else
  {
    mode = getSignal() ? -1 : 1;
  }

  // update label state
  if ((is_stable && abs(t_diff) <= MAX_TEMP_DELTA) || abs(t_diff) <= MAX_TEMP_DELTA / 2)
  {
    if (millis() > time_of_last_instability + 60 * 1000)
    {
      // STATE 3: Print "Ready"
      label_state = 3;
    }
    else if (millis() > time_of_last_instability + 30 * 1000) // if (report_when_ready)
    {
      // STATE 2: Print "Ready in {X}" seconds
      label_state = 2;
    }
    else
    {
      // STATE 1: Print "Wait for Ready"
      label_state = 1;
    }
  }
  else
  {
    // STATE 0: Print "Temp Cycling"
    label_state = 0;

    time_of_last_instability = millis();
  }

  bool debug = false;
  if (debug)
  {
    Serial.print("target: ");
    Serial.print(target);
    Serial.print("\tactual: ");
    Serial.print(t);
    Serial.print("\tambient: ");
    Serial.print(ambient);
    Serial.print("\t reference: ");
    Serial.print(reference);
    Serial.print("\tPID_error: ");
    Serial.print(PID_error);
    Serial.print("  kp: ");
    Serial.print(kp);
    Serial.print("  ki: ");
    Serial.print(ki);
    Serial.print("  kd: ");
    Serial.print(kd);
    Serial.print("  \tPID_p: ");
    Serial.print(PID_p);
    Serial.print("\tPID_i: ");
    Serial.print(PID_i);
    Serial.print("\tPID_d: ");
    Serial.print(PID_d);
    Serial.print("\tpower: ");
    Serial.print(power);
    Serial.println();
  }

  return mode;
}

void L298NHB::shutdown(void)
{
  setPower(0); // off
  target = ambient = 25.0;
}

void L298NHB::wakeup(void)
{
  setPower(e0); // init on
  // setSignal(m0);
  time_of_last_instability = millis(); // not zero, but time of TEC start
  label_state = 0;                     // STATE 0: Print "Temp Cycling"
}

bool L298NHB::active(void)
{
  return (power != 0);
}
