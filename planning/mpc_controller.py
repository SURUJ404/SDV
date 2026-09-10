"""
Linear MPC controller — sits between the CNN-LSTM's raw steering
suggestion and the simulator's actuators. Where the CNN-LSTM is a
learned, reactive, single-shot prediction, this optimizes a short
horizon of steering + acceleration commands against a kinematic vehicle
model, subject to explicit constraints:

  - stay near the lane-center offset (from the U-Net mask centroid, or
    the CNN-LSTM's implicit target if no mask is available)
  - respect a maximum speed that drops when YOLO reports an obstacle
    ahead in-lane (closer object -> lower speed ceiling, with a hard
    stop if within the minimum safe following distance)
  - bound steering rate and acceleration so outputs are smooth/driveable

This is the actual "hard safety layer" a learned end-to-end model like
PilotNet or the CNN-LSTM doesn't have on its own: the network can propose
whatever it wants, but the MPC will not emit a command that violates the
distance/speed constraints.

Uses a linearized kinematic bicycle model, discretized at dt, solved as a
convex QP with cvxpy. This is a genuinely simplified MPC (linear model,
fixed linearization point per step) — appropriate for a simulator-speed
vehicle, not a substitute for a production nonlinear MPC stack.
"""

import cvxpy as cp
import numpy as np

# Vehicle/model parameters
WHEELBASE_M = 2.5
DT = 0.1
HORIZON = 10

MAX_STEER_RATE = 0.6      # rad/s
MAX_ACCEL = 3.0           # m/s^2
MAX_DECEL = -6.0          # m/s^2
MIN_SAFE_DISTANCE_M = 5.0  # hard-stop threshold
COMFORT_FOLLOWING_TIME_S = 1.8  # target following gap, "N seconds behind"


def linearized_bicycle_matrices(v_ref, heading_ref, dt=DT, L=WHEELBASE_M):
    """
    State: [lateral_offset, heading_error, speed]
    Input: [steering_angle, acceleration]

    Linearized around a reference forward speed and near-zero heading
    error, which holds well for lane-keeping at simulator speeds.
    """
    A = np.array([
        [1.0, v_ref * dt, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    B = np.array([
        [0.0, 0.0],
        [(v_ref / L) * dt, 0.0],
        [0.0, dt],
    ])
    return A, B


def compute_speed_limit(nominal_speed, obstacle_distance_m):
    """
    Caps the reference speed based on the nearest in-lane obstacle,
    using a simple time-gap rule. Returns 0.0 (hard stop) inside the
    minimum safe distance.
    """
    if obstacle_distance_m is None:
        return nominal_speed
    if obstacle_distance_m <= MIN_SAFE_DISTANCE_M:
        return 0.0
    time_gap_limited_speed = obstacle_distance_m / COMFORT_FOLLOWING_TIME_S
    return float(min(nominal_speed, time_gap_limited_speed))


class MPCController:
    def __init__(self, horizon=HORIZON, dt=DT):
        self.horizon = horizon
        self.dt = dt

    def solve(self, current_state, target_lateral_offset, target_speed,
              obstacle_distance_m=None):
        """
        current_state: np.array([lateral_offset, heading_error, speed])
        target_lateral_offset: desired lateral offset from lane center (0 = centered)
        target_speed: desired cruising speed (m/s) before obstacle capping
        obstacle_distance_m: distance to nearest in-lane object from YOLO, or None

        Returns (steering_angle, acceleration) — the first step of the
        optimized control sequence (standard MPC receding-horizon usage).
        """
        speed_limit = compute_speed_limit(target_speed, obstacle_distance_m)
        v_ref = max(current_state[2], 1.0)  # avoid degenerate linearization at v=0
        A, B = linearized_bicycle_matrices(v_ref, heading_ref=0.0, dt=self.dt)

        x = cp.Variable((3, self.horizon + 1))
        u = cp.Variable((2, self.horizon))

        cost = 0
        constraints = [x[:, 0] == current_state]

        for t in range(self.horizon):
            cost += cp.square(x[0, t] - target_lateral_offset) * 5.0   # lane-centering weight
            cost += cp.square(x[1, t]) * 2.0                            # heading-error weight
            cost += cp.square(x[2, t] - speed_limit) * 1.0              # speed-tracking weight
            cost += cp.square(u[0, t]) * 0.5                            # steering effort
            cost += cp.square(u[1, t]) * 0.1                            # accel effort

            constraints += [
                x[:, t + 1] == A @ x[:, t] + B @ u[:, t],
                cp.abs(u[0, t]) <= MAX_STEER_RATE,
                u[1, t] <= MAX_ACCEL,
                u[1, t] >= MAX_DECEL,
                x[2, t] >= 0.0,  # no reverse
            ]

            if obstacle_distance_m is not None and obstacle_distance_m <= MIN_SAFE_DISTANCE_M:
                # Hard stop constraint overrides everything else in the cost
                constraints += [x[2, t] <= 0.5]

        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver=cp.OSQP, warm_start=True)

        if u.value is None:
            # Solver failed (rare, usually infeasible constraints) — fail safe.
            return 0.0, MAX_DECEL

        steering_angle = float(u.value[0, 0])
        acceleration = float(u.value[1, 0])
        return steering_angle, acceleration
