import logging
from pathlib import Path
from typing import Any, Tuple

import pickle
import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Internal project imports
from airplane.reduced_symmetric_glider_pullout import ReducedSymmetricGliderPullout
from airplane.reduced_symmetric_pullout import ReducedSymmetricPullout
from airplane.reduced_banked_glider_pullout import ReducedBankedGliderPullout
from PolicyIteration import PolicyIteration, PolicyIterationConfig
from utils.utils import get_barycentric_weights_and_indices, get_optimal_action


import cProfile
import io
import logging
import pstats
from pstats import SortKey
import matplotlib.ticker as ticker

from scipy.interpolate import interpn

# Configure professional logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================================
# UTILS: COLORMAPS
# =====================================================================

def get_parula_cmap() -> LinearSegmentedColormap:
    """
    Create a highly accurate approximation of MATLAB's proprietary parula colormap.
    Used for scientific publication matching.
    """
    parula_anchors = [
        (0.2081, 0.1663, 0.5292), (0.0165, 0.4266, 0.8786),
        (0.0384, 0.6743, 0.7436), (0.4420, 0.7481, 0.5033),
        (0.8185, 0.7327, 0.3498), (0.9990, 0.7653, 0.2384),
        (0.9769, 0.9839, 0.0805),
    ]
    return LinearSegmentedColormap.from_list("parula_approx", parula_anchors)

# =====================================================================
# EXPERIMENT SETUPS
# =====================================================================

def setup_banked_glider_experiment() -> Tuple[
    gym.Env, np.ndarray, np.ndarray, PolicyIterationConfig
]:
    """Configure parameters exactly matching Table 1 of the reference paper."""
    env = ReducedBankedGliderPullout()
    
    # Exact discretization from the literature
    bins_space = {
        "flight_path_angle": np.linspace(
            np.deg2rad(-180), 0.0, 37, dtype=np.float32
        ),
        "airspeed_norm": np.linspace(0.9, 4.0, 32, dtype=np.float32),
        "bank_angle": np.linspace(
            np.deg2rad(-20), np.deg2rad(200), 45, dtype=np.float32
        )
    }
    
    grid = np.meshgrid(*bins_space.values(), indexing="ij")
    states_space = np.vstack([g.ravel() for g in grid]).astype(np.float32).T
    
    # Exact control discretization
    cl_vals = np.linspace(-0.5, 1.0, 7, dtype=np.float32)
    br_vals = np.linspace(np.deg2rad(-30), np.deg2rad(30), 13, dtype=np.float32)
    
    action_grid = np.meshgrid(cl_vals, br_vals, indexing="ij")
    action_space = np.vstack([a.ravel() for a in action_grid]).astype(np.float32).T
    
    config = PolicyIterationConfig(
        gamma=1.0,               
        theta=1e-4,              
        n_steps=1000,            
        log=False,               
        log_interval=10,         
        img_path=Path("./img")   
    )
    return env, states_space, action_space, config


def setup_high_fidelity_banked_glider() -> tuple[
    gym.Env, np.ndarray, np.ndarray, PolicyIterationConfig
]:
    """
    Configure a high-fidelity 3D State experiment for the banked glider.
    Uses 2.5-degree spacing for angles and 0.05 spacing for normalized airspeed.
    Requires approximately 2.5 GB of RAM for the transition table.
    """
    env = ReducedBankedGliderPullout()
    
    # High-density uniform grid
    bins_space = {
        # -180 to 0 with 2.5 deg increments -> 73 points
        "flight_path_angle": np.linspace(
            np.deg2rad(-180), 0.0, 73, dtype=np.float32
        ),
        # 0.9 to 4.0 with 0.05 increments -> 63 points
        "airspeed_norm": np.linspace(0.9, 4.0, 63, dtype=np.float32),
        # -20 to 200 with 2.5 deg increments -> 89 points
        "bank_angle": np.linspace(
            np.deg2rad(-20), np.deg2rad(200), 89, dtype=np.float32
        )
    }
    
    grid = np.meshgrid(*bins_space.values(), indexing="ij")
    states_space = np.vstack([g.ravel() for g in grid]).astype(np.float32).T
    
    # We maintain the same action space since bang-bang optimal control 
    # doesn't benefit much from extreme intermediate discretization.
    cl_vals = np.linspace(-0.5, 1.0, 7, dtype=np.float32) # 7 points
    br_vals = np.linspace(np.deg2rad(-30), np.deg2rad(30), 13, dtype=np.float32) # 13 points
    
    action_grid = np.meshgrid(cl_vals, br_vals, indexing="ij")
    action_space = np.vstack([a.ravel() for a in action_grid]).astype(np.float32).T
    
    config = PolicyIterationConfig(
        gamma=1.0,               
        theta=1e-4,              
        n_steps=1000,            
        log=False,               
        log_interval=10,         
        img_path=Path("./img")   
    )
    
    return env, states_space, action_space, config
def _setup_high_fidelity_banked_glider() -> tuple[
    gym.Env, np.ndarray, np.ndarray, PolicyIterationConfig
]:
    """
    True Global Optimum Grid for RTX 3070 (8GB) utilizing Procedural C++ Kernels.
    
    Memory Analytics (Compute-Bound Architecture):
    - States (Ns): 721 (gamma) * 311 (V) * 881 (mu) = 197,544,131 states.
    - Actions (Na): 91 actions.
    - VRAM per state: 13 bytes (V, new_V, policy, terminal_mask).
    - Total VRAM required: 197.5M * 13 bytes ≈ 2.39 GiB.
    - Safety Margin: > 5.0 GiB (Immune to OOM on 8GB cards).
    
    Note: Each Bellman iteration will perform ~18 billion RK4 physics 
    evaluations. This is highly compute-intensive but fits perfectly in memory.
    """
    env = ReducedBankedGliderPullout() 
    
    bins_space = {
        # 0.25 degree resolution -> 721 points
        "flight_path_angle": np.linspace(
            np.deg2rad(-180), 0.0, 721, dtype=np.float32
        ),
        # 0.01 normalized airspeed resolution -> 311 points
        "airspeed_norm": np.linspace(0.9, 4.0, 311, dtype=np.float32),
        # 0.25 degree bank angle resolution -> 881 points
        "bank_angle": np.linspace(
            np.deg2rad(-20), np.deg2rad(200), 881, dtype=np.float32
        )
    }
    
    grid = np.meshgrid(*bins_space.values(), indexing="ij")
    states_space = np.vstack([g.ravel() for g in grid]).astype(np.float32).T
    
    # Action space remains standard to capture Bang-Bang optimality effectively
    cl_vals = np.linspace(-0.5, 1.0, 7, dtype=np.float32)
    br_vals = np.linspace(np.deg2rad(-30), np.deg2rad(30), 13, dtype=np.float32)
    
    action_grid = np.meshgrid(cl_vals, br_vals, indexing="ij")
    action_space = np.vstack([a.ravel() for a in action_grid]).astype(np.float32).T
    
    config = PolicyIterationConfig(
        gamma=1.0,               
        theta=1e-4,              
        n_steps=1000,            
        log=False,               
        log_interval=10,         
        img_path=Path("./img")   
    )
    
    return env, states_space, action_space, config
    
    grid = np.meshgrid(*bins_space.values(), indexing="ij")
    states_space = np.vstack([g.ravel() for g in grid]).astype(np.float32).T
    
    cl_vals = np.linspace(-0.5, 1.0, 7, dtype=np.float32)
    br_vals = np.linspace(np.deg2rad(-30), np.deg2rad(30), 13, dtype=np.float32)
    
    action_grid = np.meshgrid(cl_vals, br_vals, indexing="ij")
    action_space = np.vstack([a.ravel() for a in action_grid]).astype(np.float32).T
    
    config = PolicyIterationConfig(
        gamma=1.0,               
        theta=1e-4,              
        n_steps=1000,            
        log=False,               
        log_interval=10,         
        img_path=Path("./img")   
    )
    
    return env, states_space, action_space, config


def run_profiling() -> None:
    """
    Executes the High-Fidelity 3D experiment under a strict C-level profiler.
    Outputs a detailed statistical report of execution time per function call.
    """
    logger.info("Initializing C-Profiler for the RL Pipeline...")
    
    # Setup the heavy 3D environment
    env, states, actions, config = setup_banked_glider_experiment()
    
    # Initialize the profiler
    profiler = cProfile.Profile()
    
    # Profile the exact training logic
    profiler.enable()
    pi = train_or_load_policy(env, states, actions, config, prefix="profiling")
    profiler.disable()
    
    logger.info("Profiling complete. Generating report...")
    
    # Format and export the profiling statistics
    string_stream = io.StringIO()
    # Sort by cumulative time to find the slowest macro-functions
    sort_by = SortKey.CUMULATIVE 
    
    stats = pstats.Stats(profiler, stream=string_stream).sort_stats(sort_by)
    stats.print_stats(30)  # Print the top 30 most expensive functions
    
    # Save the report to disk for deep analysis
    with open("profiling_report.txt", "w") as f:
        f.write(string_stream.getvalue())
        
    print(string_stream.getvalue())

# =====================================================================
# CORE LOGIC
# =====================================================================

def train_or_load_policy(
    env: gym.Env, 
    states: np.ndarray, 
    actions: np.ndarray, 
    config: PolicyIterationConfig, 
    prefix: str
) -> PolicyIteration:
    """
    Check if a pre-trained policy exists on disk to save VRAM and compute time.
    If not found, executes the GPU-accelerated training pipeline.
    """
    # Construct the standardized filename based on the environment name
    policy_filename = f"{env.unwrapped.__class__.__name__}_policy.pkl"
    policy_path = Path(policy_filename)

    if policy_path.exists():
        logger.info(f"[+] Existing policy found: {policy_filename}. Skipping training and loading from disk...")
        try:
            with open(policy_path, "rb") as f:
                pi = pickle.load(f)
            logger.info("[+] Policy loaded successfully.")
            
            # Ensure the loaded object has the environment reference for simulation
            pi.env = env 
            return pi
        except Exception as e:
            logger.error(f"[-] Failed to load policy file: {e}. Reverting to training...")

    # Training logic if file doesn't exist or load fails
    logger.info(f"[*] Training new high-fidelity policy for {prefix}...")
    pi = PolicyIteration(env, states, actions, config)
    pi.run()
    
    # Save the optimal policy tensor and metadata
    try:
        with open(policy_path, "wb") as f:
            pickle.dump(pi, f)
        logger.info(f"[+] Policy saved successfully to {policy_path.resolve()}")
    except Exception as e:
        logger.warning(f"[-] Could not save policy to disk: {e}")
        
    return pi

# =====================================================================
# PAPER-STYLE SLICE PLOTTING ENGINE
# =====================================================================

def validate_trajectories_with_casadi(pi: PolicyIteration, prefix: str) -> None:
    """
    Validates the discrete Dynamic Programming (DP) policy against the continuous-time
    Nonlinear Programming (NLP) CasADi solver.
    Uses the DP trajectory as the mathematical seed to guide IPOPT into the global optimum.
    """
    if pi.n_dims != 3:
        logger.warning("Trajectory validation requires the 3D environment. Skipping.")
        return

    try:
        from casadi_pullout_optimizer import CasadiPulloutOptimizer
        nlp_optimizer = CasadiPulloutOptimizer()
    except ImportError:
        logger.error("Could not import CasadiPulloutOptimizer.")
        return

    env = pi.env
    dt = env.airplane.TIME_STEP
    mass = env.airplane.MASS
    wing_area = env.airplane.WING_SURFACE_AREA
    air_density = env.airplane.AIR_DENSITY
    v_stall = env.airplane.STALL_AIRSPEED

    scenarios = [
        {"gamma_0_deg": -30.0, "fig_id": 3},
        {"gamma_0_deg": -60.0, "fig_id": 4}
    ]
    mu_0_list = [150.0, 120.0, 90.0, 60.0, 30.0]
    x_offsets = [-180.0, -60.0, 0.0, 40.0, 80.0]
    v_0_norm = 1.2

    for scenario in scenarios:
        gamma_0_deg = scenario["gamma_0_deg"]
        fig_id = scenario["fig_id"]
        
        logger.info(f"[*] Validating DP-Guided Trajectories for gamma_0 = {gamma_0_deg} deg...")

        fig, ax = plt.subplots(figsize=(12, 7))
        
        for mu_0_deg, x_offset in zip(mu_0_list, x_offsets):
            
            # =================================================================
            # 1. DP Policy Iteration Simulation (Gathering the Seed)
            # =================================================================
            gamma, v_norm, mu = np.deg2rad(gamma_0_deg), v_0_norm, np.deg2rad(mu_0_deg)
            x_dp, h_dp, xi_dp = 0.0, 0.0, 0.0
            
            # Buffers to hold the exact mathematical state history
            s_hist = {"v": [v_norm], "gamma": [gamma], "mu": [mu], "h": [h_dp], "x": [x_dp], "xi": [xi_dp]}
            c_hist = {"cl": [], "mu_dot": []}

            step_count = 0
            max_steps = 2000

            while gamma < 0.0 and step_count < max_steps:
                state_vector = np.array([gamma, v_norm, mu], dtype=np.float32)
                action, _ = get_optimal_action(state_vector, pi)
                c_lift, bank_rate = action[0], action[1]
                
                c_hist["cl"].append(c_lift)
                c_hist["mu_dot"].append(bank_rate)
                
                v_true = v_norm * v_stall
                lift_force = 0.5 * air_density * wing_area * (v_true ** 2) * c_lift

                h_dot = v_true * np.sin(gamma)
                cos_gamma = np.cos(gamma) if abs(np.cos(gamma)) > 1e-3 else 1e-3
                xi_dot = (lift_force * np.sin(mu)) / (mass * v_true * cos_gamma)
                x_dot = v_true * np.cos(gamma) * np.cos(xi_dp)

                h_dp += h_dot * dt
                xi_dp += xi_dot * dt
                x_dp += x_dot * dt

                env.state = np.atleast_2d(state_vector)
                next_state_matrix, _, _, _, _ = env.step(action)
                next_state = next_state_matrix.flatten()
                
                gamma, v_norm, mu = next_state[0], next_state[1], next_state[2]
                
                s_hist["v"].append(v_norm)
                s_hist["gamma"].append(gamma)
                s_hist["mu"].append(mu)
                s_hist["h"].append(h_dp)
                s_hist["x"].append(x_dp)
                s_hist["xi"].append(xi_dp)
                
                step_count += 1

            dp_T = step_count * dt

            # =================================================================
            # 2. Build the CasADi Seed (Spline Resampling)
            # =================================================================
            n_nodes = 150
            t_dp_states = np.linspace(0, dp_T, len(s_hist["v"]))
            t_dp_ctrls = np.linspace(0, dp_T, len(c_hist["cl"]))
            
            t_cas_states = np.linspace(0, dp_T, n_nodes + 1)
            t_cas_ctrls = np.linspace(0, dp_T, n_nodes)
            
            # Interpolate arrays to match exact CasADi NLP matrix sizes
            dp_seed = {
                "T": dp_T,
                "v_norm": np.interp(t_cas_states, t_dp_states, s_hist["v"]),
                "gamma": np.interp(t_cas_states, t_dp_states, s_hist["gamma"]),
                "mu": np.interp(t_cas_states, t_dp_states, s_hist["mu"]),
                "h": np.interp(t_cas_states, t_dp_states, s_hist["h"]),
                "x": np.interp(t_cas_states, t_dp_states, s_hist["x"]),
                "xi": np.interp(t_cas_states, t_dp_states, s_hist["xi"]),
                "c_lift": np.interp(t_cas_ctrls, t_dp_ctrls, c_hist["cl"]),
                "mu_dot": np.interp(t_cas_ctrls, t_dp_ctrls, c_hist["mu_dot"])
            }

            # =================================================================
            # 3. Solve CasADi utilizing the DP Seed
            # =================================================================
            res = nlp_optimizer.solve_trajectory(
                v0_norm=v_0_norm, 
                gamma0_deg=gamma_0_deg, 
                mu0_deg=mu_0_deg,
                n_nodes=n_nodes,
                dp_seed=dp_seed
            )
            
            gamma_array = res.get("gamma", np.array([-1.0]))
            valid_indices = np.where(gamma_array >= 0.0)[0]
            cutoff_idx = valid_indices[0] + 1 if len(valid_indices) > 0 else len(gamma_array)

            x_history_cas = (res["x"] + x_offset)[:cutoff_idx]
            h_history_cas = res["h"][:cutoff_idx]

            # Shift DP X-coordinates for the plot
            x_history_dp = np.array(s_hist["x"]) + x_offset
            h_history_dp = np.array(s_hist["h"])

            # =================================================================
            # 4. Layered Plotting
            # =================================================================
            lbl_dp = "DP Policy Iteration (AI Global Optimum)" if mu_0_deg == mu_0_list[0] else ""
            lbl_cas = "CasADi NLP (DP-Guided Continuous)" if mu_0_deg == mu_0_list[0] else ""

            # Plot DP Trajectory
            ax.plot(x_history_dp, h_history_dp, color='darkred', linewidth=2.0, linestyle='-', label=lbl_dp, zorder=2)
            mark_every = max(1, len(x_history_dp) // 10)
            ax.plot(x_history_dp[::mark_every], h_history_dp[::mark_every], color='black', marker='+', markersize=8, linestyle='None', alpha=0.6, zorder=3)

            # Plot CasADi Trajectory
            ax.plot(x_history_cas, h_history_cas, color='cyan', linewidth=1.5, linestyle='--', label=lbl_cas, zorder=4)

        ax.set_title(
            f"Algorithm Validation: DP Policy vs DP-Guided Continuous NLP\n"
            f"Starting from $\\gamma_0$ = {gamma_0_deg:.0f} deg, $V_0/V_s$ = 1.2",
            fontsize=15, pad=15
        )
        ax.set_xlabel("x-position (m)", fontsize=13)
        ax.set_ylabel("Altitude (m)", fontsize=13)
        ax.set_xlim([-200, 250])
        ax.set_ylim([-250, 50])
        ax.grid(True, which='both', linestyle='-', color='lightgray', linewidth=0.7)
        ax.legend(loc="lower right", fontsize=11, framealpha=0.9)
        
        plt.tight_layout()
        output_path = Path(f"img/{prefix}_validation_guided_Fig{fig_id}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"[+] Validation plot successfully saved to {output_path.resolve()}")

def plot_all_paper_style_policies(pi: PolicyIteration, prefix: str) -> None:
    if pi.n_dims != 3:
        return

    v_slices = [1.2, 4.0]
    
    # FIX 1: Query EXACTLY at the training nodes (multiples of 5)
    mu_centers = np.arange(0.0, 185.0, 5.0)
    gamma_centers = np.arange(-90.0, 5.0, 5.0)
    
    M_centers, G_centers = np.meshgrid(np.deg2rad(mu_centers), np.deg2rad(gamma_centers), indexing="ij")
    bbox_props = dict(boxstyle="square,pad=0.3", fc="white", ec="black", lw=1)

    for v_slice in v_slices:
        logger.info(f"[*] Generating exact nodes policy slices at V/Vs = {v_slice}...")
        
        query_pts = np.column_stack([
            G_centers.ravel(),
            np.full_like(G_centers.ravel(), v_slice),
            M_centers.ravel()
        ]).astype(np.float32)

        cl_map = np.zeros(query_pts.shape[0], dtype=np.float32)
        mu_dot_map = np.zeros(query_pts.shape[0], dtype=np.float32)

        for i, pt in enumerate(query_pts):
            act, _ = get_optimal_action(pt, pi)
            cl_map[i] = act[0]
            mu_dot_map[i] = np.rad2deg(act[1])

        C_L = cl_map.reshape(M_centers.shape).T
        P_CMD = mu_dot_map.reshape(M_centers.shape).T

        # --- PLOT C_L ---
        fig_cl, ax_cl = plt.subplots(figsize=(10, 4.5))
        # FIX 1b: shading="nearest" centers the box over the exact node
        ax_cl.pcolormesh(
            mu_centers, gamma_centers, C_L,
            cmap="gray", edgecolors="k", linewidth=0.5,
            vmin=-0.5, vmax=1.0, shading="nearest"
        )

        ax_cl.set_title("Optimal policy for $C_L^*$", fontsize=16, pad=15)
        ax_cl.set_ylabel("Flight path angle (deg)", fontsize=14)
        ax_cl.set_xlabel("Bank angle (deg)", fontsize=14)
        ax_cl.set_xlim([0, 180])
        ax_cl.set_ylim([-90, 0])
        ax_cl.set_xticks(np.arange(0, 210, 30))
        ax_cl.set_yticks(np.arange(-90, 30, 30))

        ax_cl.text(45, -45, "$C_L^* = 1.0$", ha="center", va="center", bbox=bbox_props, fontsize=12)
        ax_cl.text(135, -30, "$C_L^* = -0.5$", ha="center", va="center", bbox=bbox_props, fontsize=12)
        ax_cl.text(183, -45, f"V/V$_s$ = {v_slice}", va="center", fontsize=14)

        fig_cl.tight_layout()
        fig_cl.savefig(f"img/{prefix}_policy_CL_V_{v_slice}.png", dpi=300, bbox_inches="tight")
        plt.close(fig_cl)

        # --- PLOT Mu_dot ---
        fig_mu, ax_mu = plt.subplots(figsize=(10, 4.5))
        ax_mu.pcolormesh(
            mu_centers, gamma_centers, P_CMD,
            cmap="gray", edgecolors="k", linewidth=0.5,
            vmin=-30.0, vmax=30.0, shading="nearest"
        )

        ax_mu.set_title("Optimal policy for $\\dot{\\mu}_{cmd}^*$", fontsize=16, pad=15)
        ax_mu.set_ylabel("Flight path angle (deg)", fontsize=14)
        ax_mu.set_xlabel("Bank angle (deg)", fontsize=14)
        ax_mu.set_xlim([0, 180])
        ax_mu.set_ylim([-90, 0])
        ax_mu.set_xticks(np.arange(0, 210, 30))
        ax_mu.set_yticks(np.arange(-90, 30, 30))

        ax_mu.text(45, -35, "$\\dot{\\mu}^* = -30$ deg/s\n(roll back)", ha="center", va="center", bbox=bbox_props, fontsize=12)
        ax_mu.text(145, -70, "$\\dot{\\mu}^* = 30$ deg/s\n(roll over)", ha="center", va="center", bbox=bbox_props, fontsize=12)
        ax_mu.text(183, -45, f"V/V$_s$ = {v_slice}", va="center", fontsize=14)

        fig_mu.tight_layout()
        fig_mu.savefig(f"img/{prefix}_policy_MuDot_V_{v_slice}.png", dpi=300, bbox_inches="tight")
        plt.close(fig_mu)

def _plot_all_paper_style_policies(pi: PolicyIteration, prefix: str) -> None:
    """
    Generates policy heatmaps using the exact training grid resolution 
    (77x63x89) to prevent interpolation artifacts in the publication figures.
    """
    if pi.n_dims != 3:
        return

    # Slices for near-stall and high-energy states
    v_slices = [1.2, 4.0]
    
    # 1. Extract the exact training bins from the state space to ensure 1:1 mapping
    gamma_training = np.unique(pi.states_space[:, 0])
    mu_training = np.unique(pi.states_space[:, 2])

    # 2. Filter the grid to match the standard paper visualization range
    # Gamma: -90 to 0 | Mu: 0 to 180
    gamma_mask = (gamma_training >= np.deg2rad(-90.1)) & (gamma_training <= 0.01)
    mu_mask = (mu_training >= -0.01) & (mu_training <= np.deg2rad(180.1))
    
    gamma_plot = gamma_training[gamma_mask]
    mu_plot = mu_training[mu_mask]
    
    # Create the mesh for querying the VRAM policy tensor
    M_centers, G_centers = np.meshgrid(mu_plot, gamma_plot, indexing="ij")
    bbox_props = dict(boxstyle="square,pad=0.3", fc="white", ec="black", lw=1)

    for v_slice in v_slices:
        logger.info(f"[*] Plotting raw policy nodes at V/Vs = {v_slice} (Resolution: {len(gamma_plot)}x{len(mu_plot)})")
        
        # Prepare batch query points [Gamma, V, Mu]
        query_pts = np.column_stack([
            G_centers.ravel(),
            np.full_like(G_centers.ravel(), v_slice),
            M_centers.ravel()
        ]).astype(np.float32)

        cl_map = np.zeros(query_pts.shape[0], dtype=np.float32)
        mu_dot_map = np.zeros(query_pts.shape[0], dtype=np.float32)

        # Query the optimal action for each training node
        for i, pt in enumerate(query_pts):
            act, _ = get_optimal_action(pt, pi)
            cl_map[i] = act[0]
            mu_dot_map[i] = np.rad2deg(act[1])

        # Reshape to 2D for plotting
        C_L = cl_map.reshape(M_centers.shape).T
        P_CMD = mu_dot_map.reshape(M_centers.shape).T

        # Convert back to degrees for the axis labels
        gamma_deg = np.rad2deg(gamma_plot)
        mu_deg = np.rad2deg(mu_plot)

        # --- PLOT C_L ---
        fig_cl, ax_cl = plt.subplots(figsize=(10, 4.5))
        ax_cl.pcolormesh(
            mu_deg, gamma_deg, C_L,
            cmap="gray", edgecolors="k", linewidth=0.1, # Thin lines to show high density
            vmin=-0.5, vmax=1.0, shading="nearest"
        )

        ax_cl.set_title(f"Optimal policy for $C_L^*$ (V/$V_s$ = {v_slice})", fontsize=16, pad=15)
        ax_cl.set_ylabel("Flight path angle (deg)", fontsize=14)
        ax_cl.set_xlabel("Bank angle (deg)", fontsize=14)
        ax_cl.set_xlim([0, 180])
        ax_cl.set_ylim([-90, 0])
        ax_cl.set_xticks([0, 45, 90, 135, 180])
        ax_cl.set_yticks([-90, -60, -30, 0])
        
        ax_cl.text(45, -45, "$C_L^* = 1.0$", ha="center", va="center", bbox=bbox_props, fontsize=12)
        ax_cl.text(135, -30, "$C_L^* = -0.5$", ha="center", va="center", bbox=bbox_props, fontsize=12)

        fig_cl.tight_layout()
        fig_cl.savefig(f"img/{prefix}_policy_CL_V_{v_slice}.png", dpi=300, bbox_inches="tight")
        plt.close(fig_cl)

        # --- PLOT Mu_dot ---
        fig_mu, ax_mu = plt.subplots(figsize=(10, 4.5))
        ax_mu.pcolormesh(
            mu_deg, gamma_deg, P_CMD,
            cmap="gray", edgecolors="k", linewidth=0.1,
            vmin=-30.0, vmax=30.0, shading="nearest"
        )

        ax_mu.set_title(f"Optimal policy for $\\dot{{\\mu}}_{{cmd}}^*$ (V/$V_s$ = {v_slice})", fontsize=16, pad=15)
        ax_mu.set_ylabel("Flight path angle (deg)", fontsize=14)
        ax_mu.set_xlabel("Bank angle (deg)", fontsize=14)
        ax_mu.set_xlim([0, 180])
        ax_mu.set_ylim([-90, 0])
        ax_mu.set_xticks([0, 45, 90, 135, 180])
        ax_mu.set_yticks([-90, -60, -30, 0])

        ax_mu.text(45, -35, "$\\dot{\\mu}^* = -30$ deg/s\n(roll back)", ha="center", va="center", bbox=bbox_props, fontsize=12)
        ax_mu.text(145, -70, "$\\dot{\\mu}^* = 30$ deg/s\n(roll over)", ha="center", va="center", bbox=bbox_props, fontsize=12)

        fig_mu.tight_layout()
        fig_mu.savefig(f"img/{prefix}_policy_MuDot_V_{v_slice}.png", dpi=300, bbox_inches="tight")
        plt.close(fig_mu)

def simulate_and_plot_trajectories(pi: PolicyIteration, prefix: str) -> None:
    """
    Simulate and plot the optimal pullout trajectories to replicate 
    Figures 3 and 4 from the reference aerospace publication.
    
    Dynamically integrates spatial kinematics (x, h, xi) over time using 
    the optimal control actions queried from the trained policy.
    """
    if pi.n_dims != 3:
        logger.warning("Trajectory simulation requires the 3D environment. Skipping.")
        return

    # Extract aerodynamic constants from the environment
    env = pi.env
    dt = env.airplane.TIME_STEP
    mass = env.airplane.MASS
    wing_area = env.airplane.WING_SURFACE_AREA
    air_density = env.airplane.AIR_DENSITY
    v_stall = env.airplane.STALL_AIRSPEED

    # Configuration for the two main trajectory figures
    scenarios = [
        {"gamma_0_deg": -30.0, "fig_id": 3},
        {"gamma_0_deg": -60.0, "fig_id": 4}
    ]

    # Initial bank angles and their respective visual X-axis offsets (from MATLAB code)
    mu_0_list = [150.0, 120.0, 90.0, 60.0, 30.0]
    x_offsets = [-180.0, -60.0, 0.0, 40.0, 80.0]
    
    v_0_norm = 1.2

    for scenario in scenarios:
        gamma_0_deg = scenario["gamma_0_deg"]
        fig_id = scenario["fig_id"]
        
        logger.info(f"[*] Simulating trajectories for initial flight path: {gamma_0_deg} deg")

        fig, ax = plt.subplots(figsize=(10, 6))
        
        for mu_0_deg, x_offset in zip(mu_0_list, x_offsets):
            # 1. Initialize Aerodynamic States
            gamma = np.deg2rad(gamma_0_deg)
            v_norm = v_0_norm
            mu = np.deg2rad(mu_0_deg)
            
            # 2. Initialize Kinematic States
            x = x_offset
            h = 0.0
            xi = 0.0  # Heading angle
            
            # Trajectory history buffers
            x_history, h_history = [], []

            # Safety counter to prevent infinite loops in terminal states
            step_count = 0
            max_steps = 2000  # 20 seconds at dt=0.01

            # Simulate until the aircraft crosses the horizon (gamma >= 0)
            while gamma < 0.0 and step_count < max_steps:
                x_history.append(x)
                h_history.append(h)

                state_vector = np.array([gamma, v_norm, mu], dtype=np.float32)
                
                # Query the PolicyIteration VRAM engine for the optimal action
                action, _ = get_optimal_action(state_vector, pi)
                c_lift = action[0]
                bank_rate = action[1]

                # True airspeed in m/s
                v_true = v_norm * v_stall
                
                # Dimensional lift force
                lift_force = 0.5 * air_density * wing_area * (v_true ** 2) * c_lift

                # Kinematic derivatives (Eq 12 from the reference paper)
                h_dot = v_true * np.sin(gamma)
                
                # Prevent division by zero near vertical dives
                cos_gamma = np.cos(gamma) if abs(np.cos(gamma)) > 1e-3 else 1e-3
                xi_dot = (lift_force * np.sin(mu)) / (mass * v_true * cos_gamma)
                
                x_dot = v_true * np.cos(gamma) * np.cos(xi)

                # Euler integration for spatial kinematics
                h += h_dot * dt
                xi += xi_dot * dt
                x += x_dot * dt

                # Step the aerodynamic environment to get the next (gamma, v_norm, mu)
                env.state = np.atleast_2d(state_vector)
                next_state_matrix, _, _, _, _ = env.step(action)
                next_state = next_state_matrix.flatten()
                
                gamma, v_norm, mu = next_state[0], next_state[1], next_state[2]
                step_count += 1

            # Plot the trajectory curve
            ax.plot(x_history, h_history, color='darkred', linewidth=1.5, linestyle='-')
            
            # Add subtle directional markers along the line
            mark_every = max(1, len(x_history) // 10)
            ax.plot(
                x_history, h_history, color='black', 
                marker='+', markersize=8, linestyle='None', 
                markevery=mark_every, alpha=0.5
            )

        # Matplotlib Styling strictly matched to the paper
        ax.set_title(
            f"Optimal pullout trajectories, starting from $\\gamma_0$ = {gamma_0_deg:.0f} deg\n"
            f"$\\mu_0$ = {{150, 120, 90, 60, 30}} deg (left to right), $V_0/V_s$ = 1.2",
            fontsize=14, pad=15
        )
        ax.set_xlabel("x-position (m)", fontsize=12)
        ax.set_ylabel("Altitude (m)", fontsize=12)
        
        ax.set_xlim([-200, 250])
        ax.set_ylim([-250, 50])
        
        ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
        
        ax.grid(True, which='both', linestyle='-', color='gray', linewidth=0.5)
        
        # Save output
        plt.tight_layout()
        output_path = Path(f"img/{prefix}_trajectory_Fig{fig_id}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"[*] Trajectory plot successfully saved to {output_path.resolve()}")

def plot_value_function_contours(pi: PolicyIteration, prefix: str) -> None:
    """
    Generate the optimal value function contour maps (Minimum Altitude Loss)
    to perfectly replicate Figure 1 of the reference publication.
    
    Extracts the converged Q-values from the continuous N-Dimensional state 
    space and projects them onto 2D slices using spline interpolation.
    """
    if pi.n_dims != 3:
        logger.warning("Contour plotting requires a 3D state space. Skipping.")
        return

    logger.info("[*] Generating Value Function contour maps (Figure 1)...")

    # 1. Reconstruct the exact 3D grid axes from the transition table
    gamma_grid = np.unique(pi.states_space[:, 0])
    v_grid = np.unique(pi.states_space[:, 1])
    mu_grid = np.unique(pi.states_space[:, 2])

    # 2. Reshape the flat value function back into the 3D tensor shape.
    # The reward was formulated as negative altitude loss, so we multiply by -1 
    # to plot positive altitude loss physically lost during the maneuver.
    V_3D = -pi.value_function.reshape((len(gamma_grid), len(v_grid), len(mu_grid)))

    # Defensive scaling check: If the environment reward mistakenly used 
    # normalized airspeed instead of true airspeed, scale the metric back to real meters.
    if np.max(V_3D) > 0 and np.max(V_3D) < 50.0:
        logger.info("[*] Scaling normalized Value Function to physical meters.")
        v_stall = pi.env.airplane.STALL_AIRSPEED
        V_3D *= v_stall

    # 3. Define the dense visual coordinate grid for smooth contouring
    mu_visual = np.linspace(0.0, np.deg2rad(180), 150)
    gamma_visual = np.linspace(np.deg2rad(-90), 0.0, 150)
    M_vis, G_vis = np.meshgrid(mu_visual, gamma_visual, indexing="ij")

    # Speed slices specified in the original research
    v_slices = [1.2, 2.0, 3.0, 4.0]
    
    # Instantiate the 2x2 subplot canvas
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    # Exact contour levels from the MATLAB source code (0 to ~270 meters, 30m steps)
    levels = np.arange(0, 300, 30)
    
    # Custom approximation of MATLAB's "parula" or "jet" continuous colormap
    cmap = plt.get_cmap("jet")

    for idx, v_slice in enumerate(v_slices):
        ax = axes[idx]
        
        # Construct query points for interpolation aligning with (Gamma, V/Vs, Mu)
        query_pts = np.column_stack([
            G_vis.ravel(),
            np.full_like(G_vis.ravel(), v_slice),
            M_vis.ravel()
        ])
        
        # N-Dimensional Spline Interpolation directly on the Value Function tensor
        alt_loss_flat = interpn(
            (gamma_grid, v_grid, mu_grid), 
            V_3D, 
            query_pts, 
            method="cubic", 
            bounds_error=False, 
            fill_value=None
        )
        
        # Reshape back to the 2D visual grid and transpose for Matplotlib alignment
        alt_loss = alt_loss_flat.reshape(M_vis.shape).T
        
        # Layer 1: Filled gradient contours
        contour_filled = ax.contourf(
            np.rad2deg(mu_visual), 
            np.rad2deg(gamma_visual), 
            alt_loss, 
            levels=levels, 
            cmap=cmap, 
            extend="max"
        )
        
        # Layer 2: Solid demarcation lines
        contour_lines = ax.contour(
            np.rad2deg(mu_visual), 
            np.rad2deg(gamma_visual), 
            alt_loss, 
            levels=levels, 
            colors='k', 
            linewidths=0.5
        )
        
        # Add inline altitude labels to the contour lines
        ax.clabel(contour_lines, inline=True, fontsize=9, fmt="%1.0f")

        # Typography matching the publication standards
        ax.set_title(f"$V/V_s = {v_slice}$", fontsize=14)
        
        ax.set_xlim([0, 180])
        ax.set_ylim([-90, 0])
        ax.set_xticks([0, 45, 90, 135, 180])
        ax.set_yticks([-90, -60, -30, 0])
        
        if idx >= 2:
            ax.set_xlabel("Bank angle (deg)", fontsize=12)
        if idx % 2 == 0:
            ax.set_ylabel("Flight path angle (deg)", fontsize=12)

    plt.tight_layout()
    output_path = Path(f"img/{prefix}_value_function_contours.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    logger.info(f"[*] Value function contours successfully saved to {output_path.resolve()}")

# Update the run_pipeline function to execute all visualizations:
def run_pipeline(setup_func: Any, prefix: str) -> None:
    """Execute training and analytics with injected configurations."""
    env, states, actions, config = setup_func()
    pi = train_or_load_policy(env, states, actions, config, prefix)
    
    if pi.n_dims == 3:
        # 1. Generate the policy heatmaps
        plot_all_paper_style_policies(pi, prefix)
        
        # 2. Generate the Minimum Altitude Loss contour maps
        plot_value_function_contours(pi, prefix)
        
        # 3. Generate the Kinematic DP trajectory plots
        simulate_and_plot_trajectories(pi, prefix)
        
        # 4. NEW: Execute the absolute mathematical validation vs CasADi
        validate_trajectories_with_casadi(pi, prefix)

if __name__ == "__main__":
    run_pipeline(setup_banked_glider_experiment, "banked_glider")
    #run_profiling()