"""
Synthetic spatial scene generator for the Neural-Inspired Obstacle Salience Model.

A "scene" models a snapshot of a user navigating forward through space with
N surrounding obstacles. Each obstacle has position, velocity, and size.
Ground-truth threat labels are computed by forward-simulating the scene
and checking for near-misses within a time horizon.

Coordinate system:
    - User at origin (0, 0), facing +y direction
    - Positions in meters, velocities in m/s
    - Angles measured from heading (+y), positive = counterclockwise (left)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd


# ---------- Configuration ----------

USER_SPEED = 1.4                  # m/s, typical walking speed
TIME_HORIZON = 3.0                # seconds to look ahead for collision
SIM_DT = 0.05                     # integration timestep for forward sim
PERSONAL_SPACE = 0.8              # meters, "too close" buffer around user
MAX_DETECTION_RANGE = 10.0        # meters, sensor range cutoff


# ---------- Data structures ----------

@dataclass
class Obstacle:
    x: float           # meters, lateral offset (right positive)
    y: float           # meters, forward offset (ahead positive)
    vx: float          # m/s, lateral velocity (world frame)
    vy: float          # m/s, forward velocity (world frame)
    size: float        # meters, effective radius

    def features(self, user_speed: float = USER_SPEED) -> dict:
        """Derived features the models will see, all in user-centric frame."""
        # Relative velocity (user is moving +y at user_speed)
        rvx = self.vx
        rvy = self.vy - user_speed

        distance = float(np.hypot(self.x, self.y))
        # Angle from heading (+y axis). atan2(x, y) gives signed angle: right = positive.
        angle = float(np.arctan2(self.x, self.y))

        # Radial velocity: rate of change of distance. Negative = approaching.
        if distance > 1e-6:
            radial_v = (self.x * rvx + self.y * rvy) / distance
        else:
            radial_v = 0.0
        # Tangential speed (perpendicular to line of sight)
        tangential_v = float(np.hypot(rvx, rvy) - abs(radial_v))

        # Time-to-contact estimate (only meaningful if approaching)
        if radial_v < -1e-3:
            ttc = max(0.0, (distance - self.size - PERSONAL_SPACE) / (-radial_v))
        else:
            ttc = float("inf")

        return {
            "distance": distance,
            "angle": angle,
            "radial_velocity": radial_v,
            "tangential_velocity": tangential_v,
            "size": self.size,
            "ttc": ttc,
            # Looming rate: common SC-literature feature (angular size expansion)
            "loom_rate": (2 * self.size * max(0.0, -radial_v)) / max(distance, 0.1) ** 2,
        }


@dataclass
class Scene:
    obstacles: List[Obstacle] = field(default_factory=list)
    user_speed: float = USER_SPEED

    def to_dataframe(self) -> pd.DataFrame:
        rows = [o.features(self.user_speed) for o in self.obstacles]
        return pd.DataFrame(rows)


# ---------- Ground-truth threat labeling ----------

def min_distance_over_horizon(
    obstacle: Obstacle,
    user_speed: float = USER_SPEED,
    horizon: float = TIME_HORIZON,
    dt: float = SIM_DT,
) -> tuple[float, float]:
    """Forward-simulate a straight-line user trajectory and return
    (min_distance, t_of_min_distance) over the horizon."""
    steps = int(horizon / dt) + 1
    t = np.arange(steps) * dt
    # User position at time t: (0, user_speed * t)
    user_y = user_speed * t
    # Obstacle position at time t:
    obs_x = obstacle.x + obstacle.vx * t
    obs_y = obstacle.y + obstacle.vy * t
    dists = np.hypot(obs_x - 0.0, obs_y - user_y) - obstacle.size
    idx = int(np.argmin(dists))
    return float(dists[idx]), float(t[idx])


def label_scene(scene: Scene) -> pd.DataFrame:
    """Return features + ground-truth threat labels for each obstacle.

    Labels:
        is_threat:     min forward-sim distance enters personal space
        threat_time:   time at which min distance occurs (inf if no threat)
        threat_rank:   integer rank among threats, 1 = most urgent (smallest time)
                       Non-threats get rank = 0.
    """
    df = scene.to_dataframe()
    min_dists, t_mins = [], []
    for obs in scene.obstacles:
        d, t = min_distance_over_horizon(obs, scene.user_speed)
        min_dists.append(d)
        t_mins.append(t)
    df["min_forward_distance"] = min_dists
    df["threat_time"] = t_mins
    df["is_threat"] = df["min_forward_distance"] < PERSONAL_SPACE

    # Rank threats by time-to-collision (earlier = rank 1)
    threat_mask = df["is_threat"].values
    ranks = np.zeros(len(df), dtype=int)
    if threat_mask.any():
        threat_times = np.where(threat_mask, df["threat_time"].values, np.inf)
        order = np.argsort(threat_times)  # ascending time
        r = 1
        for idx in order:
            if threat_mask[idx]:
                ranks[idx] = r
                r += 1
    df["threat_rank"] = ranks
    return df


# ---------- Scene sampling ----------

def sample_scene(
    rng: np.random.Generator,
    n_obstacles_range: tuple[int, int] = (3, 8),
    danger_bias: float = 0.4,
) -> Scene:
    """Sample a random scene. `danger_bias` is the probability each obstacle
    is biased toward being on a near-collision course (makes datasets less
    trivially easy — forces the models to discriminate)."""
    n = rng.integers(n_obstacles_range[0], n_obstacles_range[1] + 1)
    obstacles = []
    for _ in range(n):
        if rng.random() < danger_bias:
            # Place obstacle ahead and give it a velocity that roughly intercepts
            y = rng.uniform(2.0, MAX_DETECTION_RANGE)
            x = rng.uniform(-2.0, 2.0)
            # Aim velocity roughly at user's future position
            t_guess = y / USER_SPEED
            target_x = rng.uniform(-0.3, 0.3)
            target_y = USER_SPEED * t_guess + rng.uniform(-0.5, 0.5)
            vx = (target_x - x) / t_guess + rng.normal(0, 0.3)
            vy = (target_y - y) / t_guess + rng.normal(0, 0.3)
        else:
            # Uniform in detection range, random velocity
            r = rng.uniform(0.5, MAX_DETECTION_RANGE)
            theta = rng.uniform(-np.pi, np.pi)
            x = r * np.sin(theta)
            y = r * np.cos(theta)
            vx = rng.normal(0, 1.0)
            vy = rng.normal(0, 1.0)
        size = float(rng.uniform(0.2, 0.6))
        obstacles.append(Obstacle(x=float(x), y=float(y), vx=float(vx), vy=float(vy), size=size))
    return Scene(obstacles=obstacles)


def add_sensor_noise(
    df: pd.DataFrame,
    rng: np.random.Generator,
    distance_noise_frac: float = 0.08,    # 8% relative range noise
    velocity_noise: float = 0.35,         # m/s Gaussian noise on velocity components
    angle_noise: float = 0.05,            # radians (~3 degrees)
    dropout_prob: float = 0.0,            # probability a feature reads NaN (unused for now)
) -> pd.DataFrame:
    """Corrupt observed features with realistic sensor noise.

    Ground-truth columns ('is_threat', 'threat_rank', 'threat_time', 'min_forward_distance')
    are untouched — they reflect physical reality, not sensor estimates.
    We re-derive 'ttc' and 'loom_rate' from the noised quantities so the models
    don't get a clean shortcut.
    """
    df = df.copy()
    n = len(df)

    # Noise distance proportionally, floor at 0.1m (sensors have a min range)
    d_true = df["distance"].values
    d_obs = np.maximum(0.1, d_true + rng.normal(0, distance_noise_frac * d_true, n))

    # Noise velocity components
    rv_obs = df["radial_velocity"].values + rng.normal(0, velocity_noise, n)
    tv_obs = df["tangential_velocity"].values + rng.normal(0, velocity_noise, n)

    # Noise angle
    a_obs = df["angle"].values + rng.normal(0, angle_noise, n)

    # Re-derive TTC from noised quantities
    ttc_obs = np.where(
        rv_obs < -1e-3,
        np.maximum(0.0, (d_obs - df["size"].values - PERSONAL_SPACE) / (-rv_obs)),
        1e6,
    )
    # Re-derive loom_rate from noised quantities
    loom_obs = (2 * df["size"].values * np.maximum(0.0, -rv_obs)) / np.maximum(d_obs, 0.1) ** 2

    df["distance"] = d_obs
    df["angle"] = a_obs
    df["radial_velocity"] = rv_obs
    df["tangential_velocity"] = tv_obs
    df["ttc"] = ttc_obs
    df["loom_rate"] = loom_obs
    return df


def generate_dataset(
    n_scenes: int,
    seed: int = 0,
    n_obstacles_range: tuple[int, int] = (3, 8),
    sensor_noise: bool = True,
) -> pd.DataFrame:
    """Generate a dataset of labeled scenes. Returns a long-format DataFrame
    with one row per obstacle and a scene_id column linking rows to scenes.

    If `sensor_noise` is True (default), observable features are corrupted
    with realistic sensor noise while ground-truth labels remain clean.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for i in range(n_scenes):
        scene = sample_scene(rng, n_obstacles_range=n_obstacles_range)
        df = label_scene(scene)
        df["scene_id"] = i
        # Keep raw coords too so we can visualize later (these are NOT noised)
        df["x"] = [o.x for o in scene.obstacles]
        df["y"] = [o.y for o in scene.obstacles]
        df["vx"] = [o.vx for o in scene.obstacles]
        df["vy"] = [o.vy for o in scene.obstacles]
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    # Infinite TTCs are annoying for downstream; replace with a large sentinel
    out["ttc"] = out["ttc"].replace([np.inf], 1e6)
    if sensor_noise:
        out = add_sensor_noise(out, rng)
    return out


if __name__ == "__main__":
    df = generate_dataset(n_scenes=5, seed=42)
    print(df.head(20))
    print(f"\nTotal obstacles: {len(df)}")
    print(f"Threat rate: {df['is_threat'].mean():.2%}")
