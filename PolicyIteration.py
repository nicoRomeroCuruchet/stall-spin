import pickle
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import gymnasium as gym
import numpy as np
from loguru import logger

# Graceful GPU scaling: Fallback to NumPy if CuPy is not available in the environment
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = np
    GPU_AVAILABLE = False

from utils.utils import evaluate_policy_step, get_barycentric_weights_and_indices


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

    Implements a hybrid CPU/GPU architecture. Environment dynamics are precomputed 
    on the CPU to leverage Gymnasium's flexibility, and tensors are subsequently 
    ported to GPU VRAM for O(1) massive parallel Bellman updates via CuPy.
    """

    def __init__(
        self,
        env: gym.Env,
        states_space: np.ndarray,
        action_space: np.ndarray,
        config: PolicyIterationConfig,
    ) -> None:

        self.env = env
        # Force float32 for C-contiguous memory layout and SIMD vectorization
        self.states_space = np.ascontiguousarray(states_space, dtype=np.float32)
        self.action_space = action_space
        self.config = config

        self.n_states, self.n_dims = self.states_space.shape
        self.n_actions = len(self.action_space)
        self.n_corners = 2**self.n_dims

        self._precompute_grid_metadata()
        self._allocate_tensors()

    def _precompute_grid_metadata(self) -> None:
        """Extract bounds, shape, strides, and corner offsets mathematically."""
        self.bounds_low = np.min(self.states_space, axis=0).astype(np.float32)
        self.bounds_high = np.max(self.states_space, axis=0).astype(np.float32)

        # Calculate unique bins per dimension O(N) instead of O(N log N)
        self.grid_shape = np.array(
            [len(np.unique(self.states_space[:, d])) for d in range(self.n_dims)],
            dtype=np.int32,
        )

        # Calculate flat memory strides for fast N-D to 1-D indexing
        self.strides = np.zeros(self.n_dims, dtype=np.int32)
        stride = 1
        for d in range(self.n_dims - 1, -1, -1):
            self.strides[d] = stride
            stride *= self.grid_shape[d]

        self.corner_bits = np.array(
            list(product([0, 1], repeat=self.n_dims)), dtype=np.int32
        )

        logger.info(
            f"Grid precomputed. Shape: {self.grid_shape}, Strides: {self.strides}"
        )

    def _allocate_tensors(self) -> None:
        """Allocate strictly C-contiguous CPU tensors to guarantee cache locality."""
        self.reward = np.zeros((self.n_states, self.n_actions), dtype=np.float32)
        self.lambdas = np.zeros(
            (self.n_states, self.n_actions, self.n_corners), dtype=np.float32
        )
        self.points_indexes = np.zeros(
            (self.n_states, self.n_actions, self.n_corners), dtype=np.int32
        )

        self.value_function = np.zeros(self.n_states, dtype=np.float32)
        self.new_value_function = np.zeros(self.n_states, dtype=np.float32)
        
        # FIX: Initialize as a pure deterministic policy to avoid evaluating dead actions
        self.policy = np.zeros((self.n_states, self.n_actions), dtype=np.float32)
        self.policy[:, 0] = 1.0

        self.terminal_mask, terminal_rewards = self.env.terminal(self.states_space)

        if np.isscalar(terminal_rewards):
            self.value_function[self.terminal_mask] = terminal_rewards
            self.new_value_function[self.terminal_mask] = terminal_rewards
        else:
            self.value_function[self.terminal_mask] = terminal_rewards[self.terminal_mask]
            self.new_value_function[self.terminal_mask] = terminal_rewards[self.terminal_mask]
    
    def build_transition_table(self) -> None:
        """Simulate environment transitions and precompute all interpolation data."""
        logger.info("Building highly optimized transition table on CPU...")

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
                self.corner_bits,
            )

            # Assigning to C-contiguous slices
            self.reward[:, a_idx] = rewards
            self.lambdas[:, a_idx, :] = batch_lambdas
            self.points_indexes[:, a_idx, :] = batch_indices

        logger.success("Transition table successfully built and cached.")
        
        if GPU_AVAILABLE:
            self._push_tensors_to_gpu()

    def _push_tensors_to_gpu(self) -> None:
        """Transfer CPU-computed physical arrays directly to GPU VRAM."""
        logger.info("Pushing continuous physical tensors to GPU VRAM via PCIe...")
        self.d_reward = cp.asarray(self.reward, dtype=cp.float32)
        self.d_lambdas = cp.asarray(self.lambdas, dtype=cp.float32)
        self.d_points_indexes = cp.asarray(self.points_indexes, dtype=cp.int32)
        
        self.d_value_function = cp.asarray(self.value_function, dtype=cp.float32)
        self.d_policy = cp.asarray(self.policy, dtype=cp.float32)
        self.d_terminal_mask = cp.asarray(self.terminal_mask, dtype=cp.bool_)
        logger.success("VRAM Allocation fully established.")

    def _pull_tensors_from_gpu(self) -> None:
        """Retrieve optimized tensors from GPU back to CPU for API compatibility."""
        logger.info("Retrieving converged matrices from VRAM to CPU RAM...")
        self.value_function = cp.asnumpy(self.d_value_function)
        self.policy = cp.asnumpy(self.d_policy)

    def policy_evaluation(self) -> float:
        """Route to hardware-specific policy evaluation kernel."""
        if GPU_AVAILABLE:
            return self._policy_evaluation_gpu()
        return self._policy_evaluation_cpu()

    def _policy_evaluation_gpu(self) -> float:
        """
        GPU-accelerated policy evaluation with Asynchronous Execution.
        Breaks the CPU-GPU PCIe lockstep bottleneck by batching the convergence
        checks, allowing CUDA cores to operate at maximum theoretical bandwidth.
        """
        delta = float("inf")
        
        # 1. Map the probability matrix to a 1D array of greedy indices
        d_best_actions = cp.argmax(self.d_policy, axis=1)
        state_indices = cp.arange(self.n_states)
        
        # 2. Slice strictly the dynamics of the chosen actions. 
        # Shape drops from (States, Actions, Corners) -> (States, Corners)
        d_p_idx = self.d_points_indexes[state_indices, d_best_actions, :]
        d_lmbda = self.d_lambdas[state_indices, d_best_actions, :]
        d_current_reward = self.d_reward[state_indices, d_best_actions]
        
        # HPC Optimization: Synchronization Interval
        # Forces the CPU to queue CUDA kernels asynchronously without waiting
        # for the result, dramatically reducing PCIe latency overhead.
        SYNC_INTERVAL = 25 

        for i in range(self.config.maximum_iterations):
            # Pure VRAM algebra (Dispatched asynchronously)
            d_corner_values = self.d_value_function[d_p_idx]
            d_expected_next_v = cp.sum(d_lmbda * d_corner_values, axis=1)
            d_new_v = d_current_reward + self.config.gamma * d_expected_next_v
            
            # Absorbing state masking
            d_new_v = cp.where(self.d_terminal_mask, self.d_value_function, d_new_v)
            
            # Error tracking tracked solely in VRAM
            d_delta = cp.max(cp.abs(d_new_v - self.d_value_function))
            self.d_value_function = d_new_v
            
            # -----------------------------------------------------------------
            # THE CRITICAL FIX: Block CPU and sync via PCIe ONLY periodically
            # -----------------------------------------------------------------
            if i % SYNC_INTERVAL == 0 or i == self.config.maximum_iterations - 1:
                delta = float(d_delta.get())  # <--- Synchronization isolated here
                
                if delta < self.config.theta:
                    logger.success(f"GPU Evaluation converged at step {i} with Δ={delta:.5e}")
                    return delta

        msg = f"GPU Evaluation hit max iterations ({self.config.maximum_iterations})"
        logger.warning(f"{msg} with Δ={delta:.5e}")
        return delta

    def _policy_evaluation_cpu(self) -> float:
        """CPU policy evaluation fallback using Numba JIT."""
        logger.info("Starting CPU policy evaluation...")
        delta = float("inf")
        for i in range(self.config.maximum_iterations):
            evaluate_policy_step(
                self.lambdas,
                self.points_indexes,
                self.value_function,
                self.reward,
                self.policy,
                self.config.gamma,
                self.new_value_function,
                self.terminal_mask,
            )

            delta = float(np.max(np.abs(self.new_value_function - self.value_function)))

            self.value_function, self.new_value_function = (
                self.new_value_function,
                self.value_function,
            )

            if delta < self.config.theta:
                logger.success(f"CPU Evaluation converged at step {i} with Δ={delta:.5e}")
                return delta

        logger.warning(f"CPU Evaluation hit max iterations with Δ={delta:.5e}")
        return delta

    def policy_improvement(self) -> bool:
        """Route to hardware-specific policy improvement kernel."""
        if GPU_AVAILABLE:
            return self._policy_improvement_gpu()
        return self._policy_improvement_cpu()

    def _policy_improvement_gpu(self) -> bool:
        """Greedily improve the policy inside GPU memory."""
        # Parallel transition evaluation
        d_corner_values = self.d_value_function[self.d_points_indexes]
        d_expected_next_v = cp.sum(self.d_lambdas * d_corner_values, axis=2)
        d_q_values = self.d_reward + self.config.gamma * d_expected_next_v
        
        # Extract greedy optimal action per state
        d_best_actions = cp.argmax(d_q_values, axis=1)
        
        # Regenerate deterministic policy mask
        d_new_policy = cp.zeros_like(self.d_policy)
        d_new_policy[cp.arange(self.n_states), d_best_actions] = 1.0
        
        # Fast equality boolean reduction
        policy_stable = bool(cp.array_equal(self.d_policy, d_new_policy))
        
        if not policy_stable:
            # Shift bitwise integer mapping to count distinct state flips 
            changes = int(cp.sum(self.d_policy != d_new_policy) // 2)
            logger.info(f"GPU Policy updated: {changes} states changed their optimal action.")
            
        self.d_policy = d_new_policy
        return policy_stable

    def _policy_improvement_cpu(self) -> bool:
        """CPU policy improvement fallback."""
        logger.info("Starting CPU policy improvement...")
        corner_values = self.value_function[self.points_indexes]
        expected_next_v = np.sum(self.lambdas * corner_values, axis=2)
        q_values = self.reward + self.config.gamma * expected_next_v

        best_actions = np.argmax(q_values, axis=1)

        new_policy = np.zeros_like(self.policy)
        new_policy[np.arange(self.n_states), best_actions] = 1.0

        policy_stable = np.array_equal(self.policy, new_policy)

        if not policy_stable:
            changes = np.sum(self.policy != new_policy) // 2
            logger.info(f"CPU Policy updated: {changes} states changed their optimal action.")

        self.policy = new_policy
        return policy_stable

    def run(self) -> None:
        """Execute the complete Policy Iteration architecture."""
        self.build_transition_table()

        for n in range(self.config.n_steps):
            logger.info(f"--- Iteration {n + 1}/{self.config.n_steps} ---")
            self.policy_evaluation()
            is_stable = self.policy_improvement()

            if is_stable:
                logger.success(f"Algorithm converged optimally at iteration {n + 1}.")
                break
                
        # Synchronize VRAM to CPU RAM prior to returning control
        if GPU_AVAILABLE:
            self._pull_tensors_from_gpu()

        self.save()
        self.env.close()

    def save(self, filepath: Path | None = None) -> None:
        """Serialize and save the trained model to disk securely."""
        if filepath is None:
            filepath = Path.cwd() / f"{self.env.__class__.__name__}_policy.pkl"

        # Dynamically detach GPU bindings to prevent 'TypeError: can't pickle object'
        if hasattr(self, 'd_value_function'):
            del self.d_reward, self.d_lambdas, self.d_points_indexes
            del self.d_value_function, self.d_policy, self.d_terminal_mask
            if hasattr(self, 'd_new_value_function'):
                del self.d_new_value_function

        with filepath.open("wb") as f:
            pickle.dump(self, f)

        logger.success(f"Policy saved successfully to {filepath.resolve()}")

    @classmethod
    def load(cls, filepath: Path) -> "PolicyIteration":
        """Load a saved policy instance from disk."""
        with filepath.open("rb") as f:
            instance = pickle.load(f)

        if not isinstance(instance, cls):
            raise TypeError("Loaded object is not a valid PolicyIteration instance.")

        logger.success(f"Policy loaded successfully from {filepath.resolve()}")
        return instance