#ifndef DUAL_H_BRIDGE_STEPPER_H
#define DUAL_H_BRIDGE_STEPPER_H

#include <Arduino.h>
#include <AccelStepper.h>

class DualHBridgeStepper : public AccelStepper {
public:
    DualHBridgeStepper(uint8_t dir1, uint8_t pwm1, uint8_t dir2, uint8_t pwm2, uint8_t pwmValue = 200)
        : AccelStepper(HALF4WIRE, dir1, pwm1, dir2, pwm2, true), _dir1(dir1), _pwm1(pwm1), _dir2(dir2), _pwm2(pwm2), _pwmValue(pwmValue)
    {
        pinMode(_dir1, OUTPUT);
        pinMode(_pwm1, OUTPUT);
        pinMode(_dir2, OUTPUT);
        pinMode(_pwm2, OUTPUT);
    }

    void setPwmValue(uint8_t pwmValue) {
        _pwmValue = pwmValue;
    }

protected:
    void setOutputPins(uint8_t mask) override {
        // bit 0 of the mask corresponds to _pin[0]
        // bit 1 of the mask corresponds to _pin[1]
        // bit 2 of the mask corresponds to _pin[2]
        // bit 3 of the mask corresponds to _pin[3]

        uint8_t A_state = (mask & 0b0011);
        bool A_dir = 0;
        bool A_pwm = 0; // off by default
        switch (A_state)
        {
        case 1: // backward
            A_dir = 1;
            A_pwm = _pwmValue;
                break;
        case 2: // forward
            A_dir = 0;
            A_pwm = _pwmValue;
                break;
        }

        uint8_t B_state = (mask & 0b1100) >> 2;
        bool B_dir = 0;
        bool B_pwm = 0; // off by default
        switch (B_state)
        {
        case 1: // backward
            B_dir = 1;
            B_pwm = _pwmValue;
                break;
        case 2: // forward
            B_dir = 0;
            B_pwm = _pwmValue;
                break;
        }
        
        digitalWrite(_dir1, A_dir);
        digitalWrite(_dir2, B_dir);

        analogWrite(_pwm1, A_pwm);
        analogWrite(_pwm2, B_pwm);
    }

private:
    uint8_t _dir1, _pwm1, _dir2, _pwm2;
    uint8_t _pwmValue;
};

#endif // DUAL_H_BRIDGE_STEPPER_H
