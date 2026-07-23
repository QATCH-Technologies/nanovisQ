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
    
    // Initialize at last known position
    write(_currentAngle); 

    _attached = true;    

    return 1;
}

void POGOServo::detach() {
    if (_attached && _pin >= 0) {
        analogWrite(_pin, 0);          // Stop sending PWM pulses
        pinMode(_pin, INPUT);          // Revert pin to high-impedance state
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
    
    // Constrain input to valid degree bounds
    if (value < 0) value = 0;
    if (value > 180) value = 180;
    _currentAngle = value;
    
    // Map degrees directly to target microsecond timings
    _targetUs = map(value, 0, 180, _minUs, _maxUs);
    writeMicroseconds(_targetUs);
}

void POGOServo::writeMicroseconds(int value) {
    if (!_attached) return;
    
    // Safety clamp to protect the servo from over-travel physical damage
    if (value < 500) value = 500;
    if (value > 2500) value = 2500;
    
    // Convert time duration to hardware timer register ticks and write
    uint32_t ticks = usToTicks(value);
    
    // FIX: Prevent 8-bit dropout. 
    // If ticks drop to 7 or lower, the hardware turns the signal OFF (0V),
    // which causes the servo to whip around. Capping at 8 ticks (~625us) keeps it stable.
    if (ticks < 8) {
        ticks = 8; 
    }
    
    // Write out the 12-bit value safely (restore prior resolution)
    const uint32_t restoreResolution = analogWriteResolution(12);
    analogWrite(_pin, ticks);
    analogWriteResolution(restoreResolution);
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
