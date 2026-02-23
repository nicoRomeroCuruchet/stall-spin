import pickle
from pathlib import Path
from typing import Optional
from itertools import product
from dataclasses import dataclass

import numpy as np
import gymnasium as gym
from loguru import logger

from utils.utils import get_barycentric_weights_and_indices, evaluate_policy_step


from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class PolicyIterationConfig:
    """
    Configuration parameters for the Policy Iteration algorithm.
    Includes optimization thresholds and logging configurations.
    """
    maximum_iterations: int = 10_000
    gamma: float = 0.99
    theta: float = 1e-4
    n_steps: int = 100
    log: bool = False
    log_interval: int = 150
    img_path: Path = field(default_factory=lambda: Path("./img"))


class PolicyIteration:
    """
    High-performance Policy Iteration for continuous state spaces.
    Uses O(1) mathematical barycentric interpolation and zero-allocation
    Bellman updates via Numba for extreme scalability.
    """

    def __init__(self, env: gym.Env, states_space: np.ndarray, 
                 action_space: np.ndarray, config: PolicyIterationConfig):
        
        self.env = env
        # Force float32 for C-contiguous memory layout and SIMD vectorization
        self.states_space = np.ascontiguousarray(states_space, dtype=np.float32)
        self.action_space = action_space
        self.config = config
        
        self.n_states, self.n_dims = self.states_space.shape
        self.n_actions = len(self.action_space)
        self.n_corners = 2 ** self.n_dims
        
        self._precompute_grid_metadata()
        self._allocate_tensors()
        
    def _precompute_grid_metadata(self) -> None:
        """Extracts bounds, shape, strides, and corner offsets purely mathematically in O(N)."""
        self.bounds_low = np.min(self.states_space, axis=0).astype(np.float32)
        self.bounds_high = np.max(self.states_space, axis=0).astype(np.float32)
        
        # Calculate unique bins per dimension O(N) instead of O(N log N)
        self.grid_shape = np.array(
            [len(np.unique(self.states_space[:, d])) for d in range(self.n_dims)], 
            dtype=np.int32
        )
        
        # Calculate flat memory strides for fast N-D to 1-D indexing
        self.strides = np.zeros(self.n_dims, dtype=np.int32)
        stride = 1
        for d in range(self.n_dims - 1, -1, -1):
            self.strides[d] = stride
            stride *= self.grid_shape[d]
            
        self.corner_bits = np.array(
            list(product([0, 1], repeat=self.n_dims)), 
            dtype=np.int32
        )
        
        logger.info(f"Grid precomputed. Shape: {self.grid_shape}, Strides: {self.strides}")

    def _allocate_tensors(self) -> None:
        """Allocates strictly C-contiguous tensors to guarantee cache locality for Numba."""
        self.reward = np.zeros((self.n_states, self.n_actions), dtype=np.float32)
        self.lambdas = np.zeros((self.n_states, self.n_actions, self.n_corners), dtype=np.float32)
        self.points_indexes = np.zeros((self.n_states, self.n_actions, self.n_corners), dtype=np.int32)
        
        self.value_function = np.zeros(self.n_states, dtype=np.float32)
        self.new_value_function = np.zeros(self.n_states, dtype=np.float32)
        self.policy = np.full((self.n_states, self.n_actions), 1.0 / self.n_actions, dtype=np.float32)
        
        self.terminal_mask, terminal_rewards = self.env.terminal(self.states_space)
        
        # FIX: Check if the environment returned a scalar or an array for terminal_rewards
        if np.isscalar(terminal_rewards):
            self.value_function[self.terminal_mask] = terminal_rewards
            self.new_value_function[self.terminal_mask] = terminal_rewards
        else:
            self.value_function[self.terminal_mask] = terminal_rewards[self.terminal_mask]
            self.new_value_function[self.terminal_mask] = terminal_rewards[self.terminal_mask]

    def build_transition_table(self) -> None:
        """Simulates environment transitions and precomputes all interpolation data."""
        logger.info("Building highly optimized transition table...")
        
        # Pre-allocate temporary batch buffers to ensure C-contiguity when assigning
        batch_lambdas = np.zeros((self.n_states, self.n_corners), dtype=np.float32)
        batch_indices = np.zeros((self.n_states, self.n_corners), dtype=np.int32)

        for a_idx, action in enumerate(self.action_space):
            self.env.state = self.states_space.copy()
            next_states, rewards, _, _, _ = self.env.step(action)
            
            # O(1) Vectorized barycentric computation
            batch_lambdas, batch_indices = get_barycentric_weights_and_indices(
                next_states.astype(np.float32), 
                self.bounds_low, 
                self.bounds_high, 
                self.grid_shape, 
                self.strides, 
                self.corner_bits
            )
            
            # Assigning to C-contiguous slices
            self.reward[:, a_idx] = rewards
            self.lambdas[:, a_idx, :] = batch_lambdas
            self.points_indexes[:, a_idx, :] = batch_indices
            
        logger.success("Transition table successfully built and cached.")
            
    def policy_evaluation(self) -> float:
        """Evaluates the current policy until the value function converges."""
        logger.info("Starting policy evaluation...")
        
        delta = float('inf')
        for i in range(self.config.maximum_iterations):
            # Zero-allocation, in-place Bellman update using Numba
            evaluate_policy_step(
                self.lambdas, 
                self.points_indexes, 
                self.value_function, 
                self.reward, 
                self.policy, 
                self.config.gamma, 
                self.new_value_function, 
                self.terminal_mask
            )
            
            # Fast vectorized calculation of the maximum error
            delta = float(np.max(np.abs(self.new_value_function - self.value_function)))
            
            # Pointer swap instead of deep copying memory (O(1) vs O(N))
            self.value_function, self.new_value_function = self.new_value_function, self.value_function
            
            if delta < self.config.theta:
                logger.success(f"Evaluation converged at step {i} with Δ={delta:.5e}")
                return delta
                
        logger.warning(f"Evaluation hit max iterations ({self.config.maximum_iterations}) with Δ={delta:.5e}")
        return delta

    def policy_improvement(self) -> bool:
        """Greedily improves the policy with respect to the current value function."""
        logger.info("Starting policy improvement...")
        
        # Vectorized calculation of Q-values based on precomputed transitions
        corner_values = self.value_function[self.points_indexes]  # Shape: (S, A, C)
        expected_next_v = np.sum(self.lambdas * corner_values, axis=2)  # Shape: (S, A)
        
        q_values = self.reward + self.config.gamma * expected_next_v
        
        best_actions = np.argmax(q_values, axis=1)
        
        # Construct new policy without massive np.eye allocation
        new_policy = np.zeros_like(self.policy)
        new_policy[np.arange(self.n_states), best_actions] = 1.0
        
        policy_stable = np.array_equal(self.policy, new_policy)
        
        if not policy_stable:
            changes = np.sum(self.policy != new_policy) // 2
            logger.info(f"Policy updated: {changes} states changed their optimal action.")
            
        self.policy = new_policy
        return policy_stable

    def run(self) -> None:
        """Executes the complete Policy Iteration algorithm."""
        self.build_transition_table()
        
        for n in range(self.config.n_steps):
            logger.info(f"--- Iteration {n+1}/{self.config.n_steps} ---")
            self.policy_evaluation()
            is_stable = self.policy_improvement()
            
            if is_stable:
                logger.success(f"Algorithm converged optimally at iteration {n+1}.")
                break
                
        self.save()
        self.env.close()

    def save(self, filepath: Optional[Path] = None) -> None:
        """Serializes and saves the trained model to disk."""
        if filepath is None:
            filepath = Path.cwd() / f"{self.env.__class__.__name__}_policy.pkl"
            
        with filepath.open('wb') as f:
            pickle.dump(self, f)
            
        logger.success(f"Policy saved successfully to {filepath.resolve()}")

    @classmethod
    def load(cls, filepath: Path) -> "PolicyIteration":
        """Loads a saved policy instance from disk."""
        with filepath.open('rb') as f:
            instance = pickle.load(f)
            
        if not isinstance(instance, cls):
            raise TypeError("Loaded object is not a valid PolicyIteration instance.")
            
        logger.success(f"Policy loaded successfully from {filepath.resolve()}")
        return instance