import numpy as np
import gymnasium
from gymnasium import spaces
from matplotlib import pyplot as plt
from airplane.reduced_grumman import ReducedGrumman
from airplane.airplane_env import AirplaneEnv


class ReducedBankedGliderPullout(AirplaneEnv):

    def __init__(self, render_mode=None):

        self.airplane = ReducedGrumman()
        super().__init__(self.airplane)
        
        # Observation space: Flight Path Angle (γ), Air Speed (V), Bank Angle (μ)
        self.observation_space = spaces.Box(np.array([np.deg2rad(-180), 0.7, np.deg2rad(-20)], np.float32), 
                                            np.array([np.deg2rad(00), 4.0, np.deg2rad(200)], np.float32), shape=(3,), dtype=np.float32) 
        # Action space: Lift Coefficient (CL), Bank Rate (μ')
        self.action_space = spaces.Box(np.array([-0.5, np.deg2rad(-30)], np.float32), np.array([1.0, np.deg2rad(30)], np.float32), shape=(2,), dtype=np.float32)

        self.state: np.ndarray | None = None

    def _get_obs(self):
        return np.vstack([self.airplane.flight_path_angle, 
                          self.airplane.airspeed_norm, 
                          self.airplane.bank_angle
                          ], 
                          dtype=np.float32).T

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):

        # Choose the initial agent's state uniformly
        [flight_path_angle, airspeed_norm, bank_angle] = np.random.uniform(self.observation_space.low, self.observation_space.high)
        self.airplane.reset(flight_path_angle, airspeed_norm, bank_angle)

        observation = self._get_obs(), {}

        return observation

    def step(self, action: list):

        self.airplane.flight_path_angle  = self.state[:,0].copy()
        self.airplane.airspeed_norm = self.state[:,1].copy()
        self.airplane.bank_angle = self.state[:,2].copy()

        action = np.clip(action, self.action_space.low, self.action_space.high)
        c_lift, bank_rate = action[0], action[1]
        init_terminal, _ = self.terminal(self.state)

        self.airplane.command_airplane(c_lift, bank_rate, 0)
        # clip the state values to the observation space
        #self.airplane.flight_path_angle = np.clip(self.airplane.flight_path_angle, self.observation_space.low[0], self.observation_space.high[0])
        #self.airplane.airspeed_norm = np.clip(self.airplane.airspeed_norm, self.observation_space.low[1], self.observation_space.high[1])
        #self.airplane.bank_angle = np.clip(self.airplane.bank_angle, self.observation_space.low[2], self.observation_space.high[2])
        # Calculate step reward: Height Loss
        # TODO: Analyze policy performance based on reward implementation.
        reward = self.airplane.TIME_STEP * self.airplane.airspeed_norm * np.sin(self.airplane.flight_path_angle) #- 1e-3 * bank_rate ** 2 
        #reward = self.airplane.TIME_STEP * self.airplane.airspeed_norm * np.sin(self.airplane.flight_path_angle)*self.airplane.STALL_AIRSPEED
        terminated, _ = self.terminal(np.vstack([self.airplane.flight_path_angle, 
                                                 self.airplane.airspeed_norm, 
                                                 self.airplane.bank_angle
                                                 ], dtype=np.float32).T) 
        
        terminated |= init_terminal
        reward = np.where(init_terminal, 0, reward)
        return self._get_obs(), reward, terminated, False, self._get_info()
    

    def one_step(self, action: list):

        self.airplane.flight_path_angle  = self.state[0].copy()
        self.airplane.airspeed_norm = self.state[1].copy()
        self.airplane.bank_angle = self.state[2].copy()

        action = np.clip(action, self.action_space.low, self.action_space.high)
        c_lift, bank_rate = action[0], action[1]
        init_terminal, _ = self.terminal_step(self.state)

        self.airplane.command_airplane(c_lift, bank_rate, 0)
        # clip the state values to the observation space
        #self.airplane.flight_path_angle = np.clip(self.airplane.flight_path_angle, self.observation_space.low[0], self.observation_space.high[0])
        #self.airplane.airspeed_norm = np.clip(self.airplane.airspeed_norm, self.observation_space.low[1], self.observation_space.high[1])
        #self.airplane.bank_angle = np.clip(self.airplane.bank_angle, self.observation_space.low[2], self.observation_space.high[2])

        # Calculate step reward: Height Loss
        # TODO: Analyze policy performance based on reward implementation.
        reward = self.airplane.TIME_STEP * self.airplane.airspeed_norm * np.sin(self.airplane.flight_path_angle) - 1e-3 * bank_rate ** 2 
        #reward = self.airplane.TIME_STEP * self.airplane.airspeed_norm * np.sin(self.airplane.flight_path_angle)*self.airplane.STALL_AIRSPEED

        terminated, _ = self.terminal_step(np.vstack([self.airplane.flight_path_angle, 
                                                 self.airplane.airspeed_norm, 
                                                 self.airplane.bank_angle], dtype=np.float32).T) 
        
        terminated |= init_terminal
        reward = np.where(init_terminal, 0, reward)
        return self._get_obs(), reward, terminated, False, self._get_info()


    def terminal_step(self, state: np.ndarray):
        state = state.squeeze()
        flight_path_angle = state[0]
        airspeed_norm = state[1]
        terminate = np.where((flight_path_angle >= np.deg2rad(0)) &\
                             #  (flight_path_angle <= np.deg2rad(-180))) &\
                                  (airspeed_norm >= 1) , True, False)
        return terminate, 0

    def terminal(self, state: np.ndarray):
        flight_path_angle = state[:,0]
        airspeed_norm = state[:,1]
        #terminate = np.where(((flight_path_angle >= np.deg2rad(0)) |\
        #                       (flight_path_angle <= np.deg2rad(-180))) &\
        #                          (airspeed_norm >= 1) , True, False)
        #terminate = np.where((flight_path_angle >= np.deg2rad(0)) &\
                             #  (flight_path_angle <= np.deg2rad(-180))) &\
        #                          (airspeed_norm >= 1) , True, False)
        
        terminate =  np.where((flight_path_angle >= 0.0) & (airspeed_norm >= 1) , True, False)
        return terminate, 0