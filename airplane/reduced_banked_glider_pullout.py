import numpy as np
from gymnasium import spaces

from airplane.airplane_env import AirplaneEnv
from airplane.reduced_grumman import ReducedGrumman


class ReducedBankedGliderPullout(AirplaneEnv):
    """
    Environment for the asymmetric (banked) glider pullout maneuver.
    State Space: 3D (Flight Path Angle, Airspeed, Bank Angle).
    Action Space: 2D (Lift Coefficient, Bank Rate).
    """

    def __init__(self, render_mode=None):
        self.airplane = ReducedGrumman()
        super().__init__(self.airplane, render_mode=render_mode)
        
        # Observation space: Flight Path Angle (γ), Air Speed (V), Bank Angle (μ)
        low_obs = np.array([np.deg2rad(-180), 0.7, np.deg2rad(-180)], dtype=np.float32)
        high_obs = np.array([0.0, 4.0, np.deg2rad(180)], dtype=np.float32)
        
        self.observation_space = spaces.Box(
            low=low_obs, high=high_obs, shape=(3,), dtype=np.float32
        )
        
        # Action space: Lift Coefficient (CL), Bank Rate (μ')
        low_action = np.array([-0.5, np.deg2rad(-30)], dtype=np.float32)
        high_action = np.array([1.0, np.deg2rad(30)], dtype=np.float32)
        
        self.action_space = spaces.Box(
            low=low_action, high=high_action, shape=(2,), dtype=np.float32
        )

        self.np_random = np.random.default_rng()

    def _get_obs(self) -> np.ndarray:
        """Retrieve the vectorized current observation."""
        return np.vstack([
            self.airplane.flight_path_angle, 
            self.airplane.airspeed_norm, 
            self.airplane.bank_angle
        ], dtype=np.float32).T

    def _get_info(self) -> dict:
        return {}

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        """Reset the environment to a uniformly sampled valid initial state."""
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        flight_path, airspeed, bank_angle = self.np_random.uniform(
            self.observation_space.low, self.observation_space.high
        )
        self.airplane.reset(flight_path, airspeed, bank_angle)

        observation = self._get_obs()
        observation = np.clip(
            observation, self.observation_space.low, self.observation_space.high
        )

        self.state = observation.copy()
        return observation.flatten(), self._get_info()

    def step(self, action: np.ndarray) -> tuple:
        """Execute a vectorized integration step for the banked physics model."""
        action_batch = np.atleast_2d(action)
        
        # Synchronize physics engine
        self.airplane.flight_path_angle = self.state[:, 0].copy()
        self.airplane.airspeed_norm = self.state[:, 1].copy()
        self.airplane.bank_angle = self.state[:, 2].copy()

        action_clipped = np.clip(action_batch, self.action_space.low, self.action_space.high)
        
        c_lift = action_clipped[:, 0]
        bank_rate = action_clipped[:, 1]
        
        init_terminal, _ = self.terminal(self.state)

        # Propagate dynamics (Thrust = 0 for glider)
        self.airplane.command_airplane(c_lift, bank_rate, 0.0)

        # Reward: Minimize altitude loss
        reward = (
            self.airplane.TIME_STEP 
            * self.airplane.airspeed_norm 
            * np.sin(self.airplane.flight_path_angle) 
            * self.airplane.STALL_AIRSPEED
        )

        obs = self._get_obs()
        terminated, _ = self.terminal(obs) 
        terminated |= init_terminal
        
        reward = np.where(init_terminal, 0.0, reward)
        self.state = obs.copy()

        # Handle Gym API compatibility for single-vector input
        if self.state.shape[0] == 1:
            return obs.flatten(), float(reward[0]), bool(terminated[0]), False, self._get_info()
            
        return obs, reward, terminated, False, self._get_info()
    
    def terminal(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate terminal conditions enforcing 3D boundaries."""
        gamma = state[:, 0]
        v_norm = state[:, 1]
        mu = state[:, 2]

        v_max = self.observation_space.high[1]
        mu_max = self.observation_space.high[2]

        is_terminal = (
            ((gamma >= 0.0) & (v_norm >= 1.0))         # Success
            | (gamma <= -np.pi)                        # Vertical dive crash
            | (v_norm > v_max)                         # Overspeed limit
            | (np.abs(mu) >= mu_max)                   # Over-bank failure
        )
        
        terminate = is_terminal.astype(np.bool_)
        terminal_rewards = np.zeros_like(terminate, dtype=np.float32)

        return terminate, terminal_rewards