import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EpisodeRecord:
    """Stores per-episode telemetry for post-hoc analysis."""
    trajectory: List[np.ndarray] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    headings: List[float] = field(default_factory=list)
    concentrations: List[float] = field(default_factory=list)
    run_tumble_events: List[int] = field(default_factory=list)
    steps_taken: int = 0
    success: bool = False
    final_distance: float = float("inf")

    @property
    def cumulative_reward(self) -> float:
        return float(np.sum(self.rewards))

    @property
    def path_length(self) -> float:
        if len(self.trajectory) < 2:
            return 0.0
        pts = np.array(self.trajectory)
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

    @property
    def straight_line_distance(self) -> float:
        if len(self.trajectory) < 2:
            return 0.0
        return float(np.linalg.norm(self.trajectory[-1] - self.trajectory[0]))

    @property
    def chemotaxis_index(self) -> float:
        """
        Standard CI = (N_up - N_down) / (N_up + N_down) where N_up/down are
        the number of steps moving toward / away from food.  Range [-1, 1].
        """
        if len(self.rewards) < 2:
            return 0.0
        diffs = np.array(self.rewards[1:])   # skip the first (no prior)
        n_up   = int(np.sum(diffs > 0))
        n_down = int(np.sum(diffs < 0))
        denom  = n_up + n_down
        return (n_up - n_down) / denom if denom > 0 else 0.0

class FoodSource:
    """A single Gaussian concentration source."""
    def __init__(self, position: Tuple[float, float], strength: float = 1.0,
                 decay: float = 30.0):
        self.position = np.array(position, dtype=float)
        self.strength = strength
        self.decay    = decay

    def concentration_at(self, point: np.ndarray) -> float:
        d = np.linalg.norm(point - self.position)
        return self.strength * np.exp(-d / self.decay)

    def gradient_at(self, point: np.ndarray) -> np.ndarray:
        diff = self.position - point
        d    = np.linalg.norm(diff)
        if d < 1e-9:
            return np.zeros(2)
        c = self.concentration_at(point)
        return (c / self.decay) * (diff / d)

def total_concentration(sources: List[FoodSource], point: np.ndarray) -> float:
    return float(np.sum([s.concentration_at(point) for s in sources]))

# ---------------------------------------------------------------------------
# Main environment
# ---------------------------------------------------------------------------

class ChemotaxisEnvironment:
    def __init__(self, sensory_neurons, motor_neurons, world_size: float = 200.0, food_sources: Optional[List[FoodSource]] = None,
        success_radius: float = 10.0, max_steps: int = 10_000,
        base_speed: float = 1.0, base_tumble_rate: float = 0.04, tumble_bias_k: float = 40.0,
        sensor_noise: float = 0.01, motor_noise:  float = 0.01,
        gradient_reward_scale: float = 500.0, wall_penalty: float = -50.0, food_terminal_reward: float = 500.0,
        seed: Optional[int] = None):

        self.sensory_neurons = sensory_neurons
        self.motor_neurons   = motor_neurons
        self.world_size      = world_size
        self.success_radius  = success_radius
        self.max_steps       = max_steps

        self.base_speed        = base_speed
        self.base_tumble_rate  = base_tumble_rate
        self.tumble_bias_k     = tumble_bias_k

        self.sensor_noise = sensor_noise
        self.motor_noise  = motor_noise

        self.gradient_reward_scale  = gradient_reward_scale
        self.wall_penalty           = wall_penalty
        self.food_terminal_reward   = food_terminal_reward

        if food_sources is None:
            food_sources = [FoodSource(position=(160.0, 160.0))]
        self.food_sources = food_sources

        self._rng = np.random.default_rng(seed)

        self._build_neuron_maps()

        self._prev_conc:    float = 0.0
        self._adapted_conc: float = 0.0

        self.episode: Optional[EpisodeRecord] = None
        self._step_count: int = 0

        self.position: np.ndarray = np.zeros(2)
        self.heading:  float      = 0.0
        self._is_running: bool    = True     # True=run, False=tumbling

        self.reset()

    def _build_neuron_maps(self):
        self.sensoryMap: Dict[str, int] = {n.neuronName: n.idx for n in self.sensory_neurons}
        self.motorMap:   Dict[str, int] = {n.neuronName: n.idx for n in self.motor_neurons}

        self.leftMotors:  List[int] = []
        self.rightMotors: List[int] = []
        for n in self.motor_neurons:
            if n.neuronName.endswith("L"):
                self.leftMotors.append(n.idx)
            elif n.neuronName.endswith("R"):
                self.rightMotors.append(n.idx)


    def reset(
        self,
        start_position: Optional[Tuple[float, float]] = None,
        start_heading:  Optional[float] = None,
    ) -> np.ndarray:
        """
        Reset the environment for a new episode.

        Parameters
        ----------
        start_position : (x, y) in world units, defaults to (40, 40) ± noise.
        start_heading  : radians, defaults to random.

        Returns
        -------
        Initial sensory observation vector.
        """
        if start_position is None:
            jitter = self._rng.uniform(-5, 5, size=2)
            self.position = np.array([40.0, 40.0]) + jitter
        else:
            self.position = np.array(start_position, dtype=float)

        self.heading = (
            self._rng.uniform(0, 2 * np.pi)
            if start_heading is None
            else float(start_heading)
        )
        self._is_running   = True
        self._step_count   = 0
        self._prev_conc    = total_concentration(self.food_sources, self.position)
        self._adapted_conc = self._prev_conc

        self.episode = EpisodeRecord()
        self.episode.trajectory.append(self.position.copy())

        obs = self._observe()
        return obs

    def step(self, motor_spikes: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        assert self.episode is not None, "Call reset() before step()."
        if self.motor_noise > 0:
            motor_spikes = motor_spikes + self._rng.normal(0, self.motor_noise, motor_spikes.shape)
        left_drive, right_drive = self._decode_motor(motor_spikes)
        conc_now   = total_concentration(self.food_sources, self.position)
        delta_conc = conc_now - self._prev_conc
        tumble_prob = self.base_tumble_rate * np.exp(-self.tumble_bias_k * delta_conc)
        tumble_prob = float(np.clip(tumble_prob, 0.0, 1.0))
        if self._rng.random() < tumble_prob:
            tumble_angle = self._rng.uniform(-np.pi, np.pi)
            self.heading += tumble_angle
            self._is_running = False
            if self.episode is not None:
                self.episode.run_tumble_events.append(self._step_count)
            speed = self.base_speed * 0.2   # nearly stationary during tumble
        else:
            self._is_running = True
            turn_rate     = (right_drive - left_drive) * 0.4
            self.heading += turn_rate
            avg_drive = (left_drive + right_drive) / 2.0
            speed = self.base_speed * max(0.3, avg_drive)
        self.heading %= (2 * np.pi)
        direction = np.array([np.cos(self.heading), np.sin(self.heading)])
        new_position = self.position + direction * speed
        hit_wall = False
        for dim in range(2):
            if new_position[dim] < 0:
                new_position[dim] = -new_position[dim]            # reflect
                self.heading = self._reflect_heading(self.heading, dim)
                hit_wall = True
            elif new_position[dim] > self.world_size:
                overshoot = new_position[dim] - self.world_size
                new_position[dim] = self.world_size - overshoot   # reflect
                self.heading = self._reflect_heading(self.heading, dim)
                hit_wall = True
        self.position = new_position
        adapt_tau           = 0.05
        self._adapted_conc  += adapt_tau * (conc_now - self._adapted_conc)
        new_conc   = total_concentration(self.food_sources, self.position)
        reward     = self.gradient_reward_scale * (new_conc - self._prev_conc)
        self._prev_conc = new_conc
        if hit_wall:
            reward += self.wall_penalty
        reached_food = False
        for src in self.food_sources:
            dist = np.linalg.norm(self.position - src.position)
            if dist < self.success_radius:
                reward += self.food_terminal_reward
                reached_food = True
                break
        self._step_count += 1
        self.episode.trajectory.append(self.position.copy())
        self.episode.rewards.append(reward)
        self.episode.headings.append(self.heading)
        self.episode.concentrations.append(new_conc)
        self.episode.steps_taken = self._step_count
        self.episode.success     = reached_food
        min_dist = min(
            np.linalg.norm(self.position - s.position)
            for s in self.food_sources
        )
        self.episode.final_distance = min_dist
        done = reached_food or self._step_count >= self.max_steps
        obs = self._observe()
        info = {
            "step":            self._step_count,
            "distance":        min_dist,
            "concentration":   new_conc,
            "delta_conc":      delta_conc,
            "tumble_prob":     tumble_prob,
            "is_running":      self._is_running,
            "hit_wall":        hit_wall,
            "left_drive":      left_drive,
            "right_drive":     right_drive,
            "curr_pos":        self.position,
        }
        return obs, reward, done, info

    def _observe(self) -> np.ndarray:
        sd = 4.0

        def tip(angle_offset):
            a = self.heading + angle_offset
            return self.position + np.array([np.cos(a), np.sin(a)]) * sd
        left_tip   = tip(+0.5)
        right_tip  = tip(-0.5)
        center_tip = tip(0.0)
        left_conc   = total_concentration(self.food_sources, left_tip)
        right_conc  = total_concentration(self.food_sources, right_tip)
        center_conc = total_concentration(self.food_sources, center_tip)
        eps = 1e-9
        log_left   = np.log1p(left_conc   / eps)
        log_right  = np.log1p(right_conc  / eps)
        log_center = np.log1p(center_conc / eps)
        current_conc_avg = (left_conc + right_conc) / 2.0
        dc_dt = current_conc_avg - self._adapted_conc
        spikes = np.zeros(len(self.sensory_neurons))
        if "ASEL" in self.sensoryMap:
            spikes[self.sensoryMap["ASEL"]] = max(0.0,  dc_dt) + max(0.0, log_left  - log_right)
        if "ASER" in self.sensoryMap:
            spikes[self.sensoryMap["ASER"]] = max(0.0, -dc_dt) + max(0.0, log_right - log_left)
        if "AWCL" in self.sensoryMap:
            spikes[self.sensoryMap["AWCL"]] = log_left
        if "AWCR" in self.sensoryMap:
            spikes[self.sensoryMap["AWCR"]] = log_right
        if "AWAL" in self.sensoryMap:
            spikes[self.sensoryMap["AWAL"]] = log_center
        if "AWAR" in self.sensoryMap:
            spikes[self.sensoryMap["AWAR"]] = log_center
        if self.sensor_noise > 0:
            spikes += self._rng.normal(0, self.sensor_noise, spikes.shape)
            spikes  = np.maximum(spikes, 0.0)
        return spikes

    def _decode_motor(self, motor_spikes: np.ndarray) -> Tuple[float, float]:
        """Average spike rate for left and right motor pools."""
        left  = float(np.mean([motor_spikes[i] for i in self.leftMotors]))  if self.leftMotors  else 0.5
        right = float(np.mean([motor_spikes[i] for i in self.rightMotors])) if self.rightMotors else 0.5
        return left, right

    @staticmethod
    def _reflect_heading(heading: float, dim: int) -> float:
        """Mirror heading component on collision with a wall."""
        if dim == 0:   
            heading = np.pi - heading
        else:         
            heading = -heading
        return float(heading % (2 * np.pi))

    @classmethod
    def single_source(cls, sensory_neurons, motor_neurons, **kwargs):
        """Default single-food-source arena (original layout)."""
        return cls(
            sensory_neurons, motor_neurons,
            food_sources=[FoodSource((160.0, 160.0))],
            **kwargs
        )

    @classmethod
    def dual_source(cls, sensory_neurons, motor_neurons, **kwargs):
        """Two food sources — tests whether the worm can choose the closer one."""
        return cls(
            sensory_neurons, motor_neurons,
            food_sources=[
                FoodSource((160.0, 160.0), strength=1.0),
                FoodSource(( 40.0, 160.0), strength=0.6),
            ],
            **kwargs
        )

    @classmethod
    def gradient_ramp(cls, sensory_neurons, motor_neurons, **kwargs):
        """
        Broad shallow gradient from a distant source — tests sustained
        gradient-climbing rather than short-range nose-following.
        """
        return cls(
            sensory_neurons, motor_neurons,
            food_sources=[FoodSource((160.0, 160.0), decay=80.0)],
            **kwargs
        )

    @classmethod
    def noisy_arena(cls, sensory_neurons, motor_neurons, **kwargs):
        """Higher noise — stress-tests robustness of the SNN."""
        return cls(
            sensory_neurons, motor_neurons,
            sensor_noise=0.05,
            motor_noise=0.05,
            **kwargs
        )

    def episode_summary(self) -> dict:
        """Return a dict of key episode statistics after an episode ends."""
        if self.episode is None:
            return {}
        ep = self.episode
        return {
            "success":              ep.success,
            "steps":                ep.steps_taken,
            "cumulative_reward":    ep.cumulative_reward,
            "final_distance":       ep.final_distance,
            "path_length":          ep.path_length,
            "chemotaxis_index":     ep.chemotaxis_index,
            "tumble_count":         len(ep.run_tumble_events),
            "tumble_rate":          len(ep.run_tumble_events) / max(ep.steps_taken, 1),
        }