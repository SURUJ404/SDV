"""
PID speed controller — replaces the one-line throttle heuristic in the
original drive.py / drive_v2.py (`throttle = 1.0 - steering^2 - speed/target`)
with an actual closed-loop controller that tracks a target speed set by
the behavior/planning layer (which can lower the target when an object is
detected ahead, or on sharp curves).
"""

import time


class PIDController:
    def __init__(self, kp, ki, kd, output_limits=(-1.0, 1.0)):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.output_min, self.output_max = output_limits
        self._integral = 0.0
        self._prev_error = None
        self._prev_time = None

    def reset(self):
        self._integral = 0.0
        self._prev_error = None
        self._prev_time = None

    def step(self, setpoint, measurement, now=None):
        now = now if now is not None else time.time()
        error = setpoint - measurement

        if self._prev_time is None:
            dt = 1e-3
        else:
            dt = max(now - self._prev_time, 1e-3)

        self._integral += error * dt
        # Basic anti-windup: clamp the integral term itself
        self._integral = max(min(self._integral, 10.0), -10.0)

        derivative = 0.0 if self._prev_error is None else (error - self._prev_error) / dt

        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        output = max(min(output, self.output_max), self.output_min)

        self._prev_error = error
        self._prev_time = now
        return output


def build_default_speed_controller():
    """Tuned by hand for the Udacity simulator's throttle response — a
    reasonable starting point, not a substitute for on-track tuning."""
    return PIDController(kp=0.15, ki=0.002, kd=0.05, output_limits=(-1.0, 1.0))
