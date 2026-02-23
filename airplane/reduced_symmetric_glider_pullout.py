import numpy as np
from gymnasium import spaces

from airplane.reduced_grumman import ReducedGrumman
from airplane.airplane_env import AirplaneEnv

class ReducedSymmetricGliderPullout(AirplaneEnv):

    def __init__(self, render_mode=None):
        self.airplane = ReducedGrumman()
        super().__init__(self.airplane)

        self.observation_space = spaces.Box(np.array([np.deg2rad(-180), 0.7], np.float32), 
                                            np.array([np.deg2rad(0), 4.0], np.float32), 
                                            shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(-0.5, 1.0, shape=(1,), dtype=np.float32)

    def _get_obs(self):
        return np.vstack([self.airplane.flight_path_angle, self.airplane.airspeed_norm], dtype=np.float32).T

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        [flight_path_angle, airspeed_norm] = np.random.uniform(self.observation_space.low, self.observation_space.high)
        self.airplane.reset(flight_path_angle, airspeed_norm, 0)
        
        observation = self._get_obs()
        observation = np.clip(observation, self.observation_space.low, self.observation_space.high)
        
        # BUG FIX 2: Inicializar el estado interno del entorno
        self.state = observation.copy()
        
        return observation.flatten(), {}

    def step(self, action: float):
        self.airplane.flight_path_angle = self.state[:,0].copy() 
        self.airplane.airspeed_norm = self.state[:,1].copy() 

        # BUG FIX 3: Clip preventivo de la acción
        c_lift = np.clip(action, self.action_space.low[0], self.action_space.high[0])
        
        init_terminal, _ = self.terminal(self.state)
        
        self.airplane.command_airplane(c_lift, 0.0, 0.0)

        reward = self.airplane.TIME_STEP * self.airplane.airspeed_norm * np.sin(self.airplane.flight_path_angle) * self.airplane.STALL_AIRSPEED
    
        obs = self._get_obs()
        terminated, _ = self.terminal(obs) 
        terminated |= init_terminal
        reward = np.where(init_terminal, 0.0, reward)
        
        # Actualizar el estado interno para el próximo step
        self.state = obs.copy()

        return obs, reward, terminated, False, self._get_info()


    def terminal(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Determines if the state is terminal based on successful leveling or boundary limits.
        
        Terminal conditions:
        1. Success: Flight path angle (gamma) >= 0.0 (leveled flight).
        2. Boundary/Failure: Flight path angle (gamma) <= -pi (-180 degrees).
        
        Args:
            state: A 2D NumPy array (N, 2) where state[:, 0] is gamma (radians).
                
        Returns:
            A tuple of (boolean_mask, zero_reward_array).
        """
        # Extract flight path angle (gamma) for all states in the batch
        gamma = state[:, 0]
        v_norm = state[:, 1]
        
        # 1. Logic: Terminate if leveled (>= 0) OR if we hit the -180 degree limit (<= -pi).
        # Parentheses are strictly required here because '|' has higher precedence than '>=' or '<='.
        is_terminal = ((gamma >= 0.0) & (v_norm >= 1.0)) | (gamma <= -np.pi)
        
        # Cast to boolean for explicit compatibility with Numba/C-wrappers
        terminate = is_terminal.astype(np.bool_)
        
        # In Policy Iteration with gamma=1.0, terminal states must return 0 reward
        # to stop cumulative altitude loss calculation at that exact moment.
        terminal_rewards = np.zeros_like(terminate, dtype=np.float32)
        
        return terminate, terminal_rewards