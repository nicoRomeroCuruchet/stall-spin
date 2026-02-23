import numpy as np
import gymnasium as gym
from gymnasium import spaces
from matplotlib import pyplot as plt


from airplane.reduced_grumman import ReducedGrumman
from airplane.airplane_env import AirplaneEnv



class ReducedSymmetricPullout(AirplaneEnv):

    def __init__(self, render_mode=None):

        self.airplane = ReducedGrumman()
        super().__init__(self.airplane)

        # Observation space: Flight Path Angle (γ), Air Speed (V)
        self.observation_space = spaces.Box(np.array([-np.pi, 0.9], np.float32), np.array([0, 4.0], np.float32), shape=(2,), dtype=np.float32) 
        # Action space: Lift Coefficient, δ_throttle
        self.action_space = spaces.Box(np.array([-0.5, 0.0], np.float32), np.array([1.0, 1.0], np.float32), shape=(2,), dtype=np.float32)  

    def _get_obs(self):
        return np.vstack([self.airplane.flight_path_angle, 
                          self.airplane.airspeed_norm], dtype=np.float32).T

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):

        # Choose the initial agent's state uniformly
        [flight_path_angle, airspeed_norm] = np.random.uniform(self.observation_space.low, self.observation_space.high)
        self.airplane.reset(flight_path_angle, airspeed_norm, 0)

        observation = self._get_obs()

        return observation

    def step(self, action: list):

        # Update state
        self.airplane.flight_path_angle = self.state[:,0].copy()
        self.airplane.airspeed_norm = self.state[:,1].copy()

        # Update state
        c_lift = action[0]
        delta_throttle = action[1]

        init_terminal, _ = self.terminal(self.state) 

        self.airplane.command_airplane(c_lift, 0, delta_throttle)

        self.airplane.airspeed_norm = np.clip(self.airplane.airspeed_norm, self.observation_space.low[1], self.observation_space.high[1])
        self.airplane.flight_path_angle = np.clip(self.airplane.flight_path_angle, self.observation_space.low[0], self.observation_space.high[0])

        # Calculate step reward: Height Loss
        reward = self.airplane.TIME_STEP * self.airplane.airspeed_norm * np.sin(self.airplane.flight_path_angle)*27.331231856346

        
        terminated, _ = self.terminal(self._get_obs()) 
        terminated |= init_terminal
        reward = np.where(init_terminal, 0, reward)

        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, False, info

    def termination(self,):
        terminate = (self.airplane.flight_path_angle >= 0.0 or self.airplane.flight_path_angle <= -np.pi)
        return terminate
    
    def terminal(self, state):
        flight_path_angle = state[:,0]
        airspeed_norm = state[:,1]
        terminate =  np.where((flight_path_angle >= 0.0) & (airspeed_norm >= 1) , False, False)
        return terminate, 0 