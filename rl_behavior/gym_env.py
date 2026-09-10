"""
A from-scratch OpenAI-Gym-style environment for autonomous-driving BEHAVIOR
decisions, built in the spirit of bark-simulator/bark-ml
(https://github.com/bark-simulator/bark-ml).

What's taken from bark-ml's design (per their README/API, not their code —
their core simulator is a C++/Bazel project that isn't something to vendor
into a Python-only project like this one):
  - The same interaction pattern: `gym.make(...)`, continuous
    action = [acceleration, steering_rate], step() -> (obs, reward, done, info)
  - A highway-style scenario: ego vehicle + several other vehicles on a
    multi-lane road
  - Other vehicles driven by the Intelligent Driver Model (IDM), which is
    what bark-ml uses for its highway-v0 environment
  - A reward built from goal progress, safety (collision/near-miss), and
    comfort (penalizing harsh accel/steering) — the same categories
    bark-ml's papers describe for behavior generation

What's different / simplified, stated plainly:
  - bark-ml's environments run on BARK's compiled C++ simulation core with
    a full semantic/geometric world model. This environment reimplements
    just the kinematic + IDM pieces in plain NumPy — a lightweight research
    environment, not a port of BARK.
  - No lane graph / map format compatibility with real BARK scenario files.

PURPOSE in this project: this is the BEHAVIOR-planning layer — "should I
slow down, hold lane, or start a lane change" — sitting conceptually above
the low-level MPC/PID controllers in planning/. It is trained and evaluated
standalone here; wiring its policy output live into drive_v3.py would mean
building the same observation (nearby-vehicle state) from the Udacity
simulator's telemetry, which the simulator does not expose out of the box
(it doesn't report other vehicles' positions) — noted as a real limitation,
not glossed over.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

# ---- Scenario parameters ----
NUM_LANES = 3
LANE_WIDTH_M = 3.7
ROAD_LENGTH_M = 500.0
NUM_TRAFFIC_VEHICLES = 5
DT = 0.2
MAX_STEPS = 300
GOAL_DISTANCE_M = 400.0

# ---- IDM parameters (standard values from Treiber et al. 2000) ----
IDM_DESIRED_SPEED = 15.0
IDM_TIME_HEADWAY = 1.5
IDM_MIN_GAP = 2.0
IDM_MAX_ACCEL = 1.5
IDM_COMFORTABLE_DECEL = 2.0
IDM_DELTA = 4.0


def idm_acceleration(v, v_lead, gap):
    """Intelligent Driver Model — same model bark-ml uses to drive traffic
    in its highway-v0 scenario."""
    gap = max(gap, 1e-2)
    delta_v = v - v_lead
    s_star = IDM_MIN_GAP + max(
        0.0, v * IDM_TIME_HEADWAY + (v * delta_v) / (2 * np.sqrt(IDM_MAX_ACCEL * IDM_COMFORTABLE_DECEL))
    )
    return IDM_MAX_ACCEL * (1 - (v / IDM_DESIRED_SPEED) ** IDM_DELTA - (s_star / gap) ** 2)


class TrafficVehicle:
    def __init__(self, s, lane, v):
        self.s = s        # longitudinal position along the road (m)
        self.lane = lane  # integer lane index
        self.v = v        # speed (m/s)

    def step(self, lead_s, lead_v, dt):
        gap = lead_s - self.s if lead_s is not None else 1e3
        lv = lead_v if lead_v is not None else self.v
        accel = idm_acceleration(self.v, lv, gap)
        self.v = max(0.0, self.v + accel * dt)
        self.s += self.v * dt


class HighwayBehaviorEnv(gym.Env):
    """
    Observation (flat vector):
        [ego_s_progress, ego_lane_offset, ego_v,
         (rel_s, rel_v, rel_lane)_for_each_of_K_nearest_traffic_vehicles]

    Action: Box([-1, 1], shape=(2,)) -> scaled to
        [acceleration (m/s^2), steering_rate (rad/s)]

    Reward:
        + progress toward goal
        - collision (large, terminates episode)
        - distance-to-nearest-vehicle penalty (safety margin)
        - control-effort penalty (comfort)
        + goal-reached bonus
    """

    metadata = {"render_modes": []}
    NUM_OBSERVED_VEHICLES = 3

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        obs_dim = 3 + 3 * self.NUM_OBSERVED_VEHICLES
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self._rng = np.random.default_rng()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.ego_s = 0.0
        self.ego_lane = NUM_LANES // 2
        self.ego_lateral_offset = 0.0  # within-lane offset, meters
        self.ego_heading = 0.0
        self.ego_v = 10.0
        self.step_count = 0

        self.traffic = [
            TrafficVehicle(
                s=self._rng.uniform(20, ROAD_LENGTH_M - 20),
                lane=self._rng.integers(0, NUM_LANES),
                v=self._rng.uniform(8, 14),
            )
            for _ in range(NUM_TRAFFIC_VEHICLES)
        ]

        return self._get_obs(), {}

    def _lead_vehicle_in_lane(self, lane, s):
        candidates = [v for v in self.traffic if v.lane == lane and v.s > s]
        if not candidates:
            return None
        return min(candidates, key=lambda v: v.s)

    def _nearest_vehicles(self, k):
        ranked = sorted(self.traffic, key=lambda v: abs(v.s - self.ego_s))
        return ranked[:k]

    def _get_obs(self):
        progress = self.ego_s / GOAL_DISTANCE_M
        obs = [progress, self.ego_lateral_offset, self.ego_v]

        nearest = self._nearest_vehicles(self.NUM_OBSERVED_VEHICLES)
        for v in nearest:
            obs.extend([v.s - self.ego_s, v.v - self.ego_v, float(v.lane - self.ego_lane)])
        while len(nearest) < self.NUM_OBSERVED_VEHICLES:
            obs.extend([1e3, 0.0, 0.0])  # padding for "no vehicle here"
            nearest.append(None)

        return np.array(obs, dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        accel = float(np.clip(action[0], -1, 1)) * IDM_MAX_ACCEL
        steering_rate = float(np.clip(action[1], -1, 1)) * 0.5  # rad/s, scaled

        # ---- Ego kinematics ----
        self.ego_v = max(0.0, self.ego_v + accel * DT)
        self.ego_heading += steering_rate * DT
        self.ego_s += self.ego_v * DT
        self.ego_lateral_offset += self.ego_v * np.sin(self.ego_heading) * DT

        # Snap to a discrete lane if drifted a full lane-width
        if abs(self.ego_lateral_offset) > LANE_WIDTH_M / 2:
            lane_shift = int(np.sign(self.ego_lateral_offset))
            self.ego_lane = int(np.clip(self.ego_lane + lane_shift, 0, NUM_LANES - 1))
            self.ego_lateral_offset -= lane_shift * LANE_WIDTH_M

        # ---- Traffic update (IDM) ----
        for v in self.traffic:
            lead = self._lead_vehicle_in_lane(v.lane, v.s)
            lead_s = lead.s if lead else None
            lead_v = lead.v if lead else None
            v.step(lead_s, lead_v, DT)
            v.s %= ROAD_LENGTH_M  # wrap around for a continuous-traffic feel

        # ---- Reward ----
        nearest = self._nearest_vehicles(1)
        min_gap = abs(nearest[0].s - self.ego_s) if nearest else 1e3
        same_lane_collision = any(
            v.lane == self.ego_lane and abs(v.s - self.ego_s) < 4.0 for v in self.traffic
        )

        reward = 0.0
        reward += self.ego_v * DT * 0.1                      # progress
        reward -= 0.01 * (accel ** 2 + steering_rate ** 2)    # comfort
        reward -= 0.05 * max(0.0, 8.0 - min_gap)              # safety margin
        terminated = False

        if same_lane_collision:
            reward -= 50.0
            terminated = True

        if self.ego_s >= GOAL_DISTANCE_M:
            reward += 20.0
            terminated = True

        truncated = self.step_count >= MAX_STEPS

        return self._get_obs(), reward, terminated, truncated, {"min_gap": min_gap}


# Registration mirrors bark-ml's `gym.make("highway-v0")` usage pattern.
gym.register(
    id="sdc-highway-v0",
    entry_point="rl_behavior.gym_env:HighwayBehaviorEnv",
    max_episode_steps=MAX_STEPS,
)


if __name__ == "__main__":
    # Mirrors the getting-started snippet from bark-ml's README.
    env = gym.make("sdc-highway-v0")
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        action = np.array([0.0, 0.0])  # placeholder: hold speed, go straight
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
    print(f"Episode finished, total_reward={total_reward:.2f}")
