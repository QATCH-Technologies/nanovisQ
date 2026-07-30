#include "POGOServo.h"

POGOServo::POGOServo() {
    _pin = -1;
    _minUs = 500;  // Standard minimum pulse width (0 degrees)
    _maxUs = 2500;  // Standard maximum pulse width (180 degrees)
    _currentAngle = 0;
    _targetUs = 0;
    _attached = false;
}

POGOServo::~POGOServo() {
    detach();
}

uint8_t POGOServo::attach(int pin) {
    return attach(pin, _minUs, _maxUs);
}

uint8_t POGOServo::attach(int pin, int minPulseUs, int maxPulseUs) {
    _pin = pin;
    _minUs = minPulseUs;
    _maxUs = maxPulseUs;

    // Force pin low immediately to squash the startup glitch
    pinMode(_pin, OUTPUT);
    digitalWrite(_pin, LOW);
    delay(10); // Let the physical logic line settle
    
    // Configure high-precision hardware PWM
    analogWriteFrequency(_pin, 50);    // 50Hz standard servo refresh rate
    
    _attached = true;

    // Initialize at last known position
    write(_currentAngle); 

    return 1;
}

void POGOServo::detach() {
    if (_attached && _pin >= 0) {

        // First, tell the timer to output a 0% duty cycle.
        // On Teensy, this forces the hardware channel to transition to 0V 
        // safely at the conclusion of its normal PWM timing sequence.
        analogWrite(_pin, 0);          
        
        // CRUCIAL STEP: Wait exactly one full PWM cycle (20ms at 50Hz)
        // This guarantees that any active high pulse has finished naturally, 
        // leaving the signal line completely flat before we alter the pin mode.
        delay(20); 

        // FIX: Force the pin to remain a hard OUTPUT clamped to 0V (LOW).
        // This instantly drains the 470pF filtering capacitor to ground,
        // preventing it from generating a trailing "phantom 0-deg pulse".
        pinMode(_pin, OUTPUT);          
        digitalWrite(_pin, LOW);  // Stop sending PWM pulses

        _attached = false;
        _pin = -1;
    }
}

void POGOServo::write(int value) {
    if (!_attached) return;
    
    // Mimic standard library: values > 200 are treated as raw microseconds
    if (value > 200) {
        writeMicroseconds(value);
        return;
    }
    
    // Map degrees directly to target microsecond timings
    setCurrentAngle(value);
    _targetUs = map(_currentAngle, 0, 180, _minUs, _maxUs);
    writeMicroseconds(_targetUs);
}

void POGOServo::writeMicroseconds(int value) {
    if (!_attached) return;
    
    // Safety clamp to protect the servo from over-travel physical damage
    if (value < 500) value = 500;
    if (value > 2500) value = 2500;
    
    // Convert time duration to hardware timer register ticks and write
    uint32_t ticks = usToTicks(value);
    
    // Write out the 12-bit value safely (restore prior resolution)
    __disable_irq();  // disable interrupts
    const uint32_t restoreResolution = analogWriteResolution(12);
    analogWrite(_pin, ticks);
    analogWriteResolution(restoreResolution);
    __enable_irq();  // enable interrupts
}

int POGOServo::readMicroseconds() {
    return _targetUs;
}

int POGOServo::read() {
    return _currentAngle;
}

bool POGOServo::attached() {
    return _attached;
}

void POGOServo::setCurrentAngle(int value) {
    // Constrain input to valid degree bounds
    if (value < 0) value = 0;
    if (value > 180) value = 180;
    _currentAngle = value;
}

// Math calculation: At 50Hz, 1 full cycle = 20,000 microseconds. 
// 12-bit depth yields 4096 discrete steps. Ticks = (us / 20000) * 4096
uint32_t POGOServo::usToTicks(int us) {
    return (uint32_t)((us * 4096) / 20000);
}
