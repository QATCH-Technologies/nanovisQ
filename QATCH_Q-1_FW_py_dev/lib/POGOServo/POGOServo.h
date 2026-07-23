#ifndef POGOSERVO_H
#define POGOSERVO_H

#include <Arduino.h>

class POGOServo {
public:
    POGOServo();
    ~POGOServo();

    // Standard Servo library functions
    uint8_t attach(int pin);
    uint8_t attach(int pin, int minPulseUs, int maxPulseUs);
    void detach();
    void write(int value);        // Handles degrees (0-180) or raw microseconds (>200)
    void writeMicroseconds(int value);
    int read();                   // Returns current angle in degrees
    int readMicroseconds();       // returns current pulse width in microseconds
    bool attached();

    // Custom POGO specific functions
    void setCurrentAngle(int value);

private:
    int _pin;
    int _minUs;
    int _maxUs;
    int _currentAngle;
    int _targetUs;
    bool _attached;

    // Helper function to map microseconds to 12-bit PWM values at 50Hz
    uint32_t usToTicks(int us);
};

#endif // POGOSERVO_H
