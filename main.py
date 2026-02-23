import logging
from pathlib import Path
from typing import Any

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

def setup_glider_experiment() -> tuple:
    env = ReducedSymmetricGliderPullout()
    bins_space = {
        "flight_path_angle": np.linspace(
            np.deg2rad(-180), np.deg2rad(0), 100, dtype=np.float32
        ),
        "airspeed_norm": np.linspace(0.7, 4.0, 100, dtype=np.float32),
    }
    grid = np.meshgrid(*bins_space.values(), indexing="ij")
    states_space = np.vstack([g.ravel() for g in grid]).astype(np.float32).T
    action_space = np.linspace(-0.5, 1.0, 20, dtype=np.float32)
    config = PolicyIterationConfig(
        gamma=0.99, theta=1e-3, n_steps=100, log=False, log_interval=50, img_path=Path("./img")
    )
    return env, states_space, action_space, config


def setup_powered_experiment() -> tuple:
    env = ReducedSymmetricPullout()
    bins_space = {
        "flight_path_angle": np.linspace(
            np.deg2rad(-180), np.deg2rad(0), 100, dtype=np.float32
        ),
        "airspeed_norm": np.linspace(0.7, 4.0, 100, dtype=np.float32),
    }
    grid = np.meshgrid(*bins_space.values(), indexing="ij")
    states_space = np.vstack([g.ravel() for g in grid]).astype(np.float32).T
    
    cl_vals = np.linspace(-0.5, 1.0, 10, dtype=np.float32)
    th_vals = np.linspace(0.0, 1.0, 10, dtype=np.float32)
    action_grid = np.meshgrid(cl_vals, th_vals, indexing="ij")
    action_space = np.vstack([a.ravel() for a in action_grid]).astype(np.float32).T
    
    config = PolicyIterationConfig(
                    gamma=0.999,             # Discount factor 
                    theta=1e-3,              # Convergence threshold 
                    n_steps=1000,            # number of iterations
                    log=False,               # Enable logging
                    log_interval=10,         # Update logging more frequently
                    img_path=Path("./img")   # Custom image save directory
                )

    return env, states_space, action_space, config


def setup_banked_glider_experiment() -> tuple[
    gym.Env, np.ndarray, np.ndarray, PolicyIterationConfig
]:
    """Configure parameters for the 3D State banked glider experiment."""
    env = ReducedBankedGliderPullout()
    
    # FIX 1: Full symmetric grid (-180 to 180) to prevent physics clamping during rolls
    bins_space = {
        "flight_path_angle": np.linspace(
            np.deg2rad(-180), np.deg2rad(0), 40, dtype=np.float32
        ),
        "airspeed_norm": np.linspace(0.7, 4.0, 40, dtype=np.float32),
        "bank_angle": np.linspace(
            np.deg2rad(-180), np.deg2rad(180), 40, dtype=np.float32
        )
    }
    
    grid = np.meshgrid(*bins_space.values(), indexing="ij")
    states_space = np.vstack([g.ravel() for g in grid]).astype(np.float32).T
    
    # Action space optimized for Bang-Bang control observation
    cl_vals = np.linspace(-0.5, 1.0, 5, dtype=np.float32)
    br_vals = np.linspace(np.deg2rad(-30), np.deg2rad(30), 7, dtype=np.float32)
    action_grid = np.meshgrid(cl_vals, br_vals, indexing="ij")
    action_space = np.vstack([a.ravel() for a in action_grid]).astype(np.float32).T
    
    config = PolicyIterationConfig(
                    gamma=0.999,             # Discount factor 
                    theta=1e-3,              # Convergence threshold 
                    n_steps=1000,            # number of iterations
                    log=False,               # Enable logging
                    log_interval=10,         # Update logging more frequently
                    img_path=Path("./img")   # Custom image save directory
                )
    return env, states_space, action_space, config

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
    """Load an existing policy or train a new one using the injected configuration."""
    filename = f"Reduced{prefix.capitalize()}_policy.pkl"
    path = Path.cwd() / filename
    
    if path.exists():
        logger.info(f"[*] Loading existing policy from {filename}")
        return PolicyIteration.load(path)
    
    logger.info(f"[*] Training new policy for {prefix}...")
    pi = PolicyIteration(env, states, actions, config)
    pi.run()
    return pi

# =====================================================================
# PAPER-STYLE SLICE PLOTTING ENGINE
# =====================================================================

def plot_paper_style_policy_slice(
    pi: PolicyIteration, prefix: str, v_slice: float = 1.2
) -> None:
    """
    Generate a precise, paper-quality discrete grid policy plot.
    
    Uses exact 5-degree centroid sampling to perfectly replicate the 
    reference document's visual rendering.
    """
    if pi.n_dims != 3:
        logger.warning("Paper style slice requested but environment is not 3D. Skipping.")
        return

    logger.info(f"[*] Generating exact paper-style policy slice at V/Vs = {v_slice}...")

    # FIX 3: Define exact 5-degree edges for the rendering mesh
    mu_edges = np.linspace(0.0, 180.0, 37)     # 36 cells = 180 deg
    gamma_edges = np.linspace(-90.0, 0.0, 19)  # 18 cells = 90 deg
    
    # Query the PolicyIteration table exactly at the centroid of each visual cell
    mu_centers = np.deg2rad(mu_edges[:-1] + 2.5)
    gamma_centers = np.deg2rad(gamma_edges[:-1] + 2.5)
    
    M_centers, G_centers = np.meshgrid(mu_centers, gamma_centers, indexing="ij")

    # Stack dimensions matching [Gamma, V/Vs, Mu]
    query_pts = np.column_stack([
        G_centers.ravel(),
        np.full_like(G_centers.ravel(), v_slice),
        M_centers.ravel()
    ]).astype(np.float32)

    actions_map = np.zeros(query_pts.shape[0], dtype=np.float32)

    # Batch inference
    for i, pt in enumerate(query_pts):
        act, _ = get_optimal_action(pt, pi)
        actions_map[i] = act[0] if isinstance(act, (np.ndarray, list)) else act

    # Reshape and transpose for matrix-to-axis alignment
    C_L = actions_map.reshape(M_centers.shape).T

    fig, ax = plt.subplots(figsize=(10, 4.5))

    # Render discrete mesh using defined edge boundaries
    mesh = ax.pcolormesh(
        mu_edges, gamma_edges, C_L,
        cmap="gray", edgecolors="k", linewidth=0.5,
        vmin=-0.5, vmax=1.0
    )

    # Typography matching
    ax.set_title("Optimal policy for $C_L^*$", fontsize=16, pad=15)
    ax.set_ylabel("Flight path angle (deg)", fontsize=14)
    ax.set_xlabel("Bank angle (deg)", fontsize=14)
    
    ax.set_xlim([0, 180])
    ax.set_ylim([-90, 0])
    
    ax.set_xticks(np.arange(0, 210, 30))
    ax.set_yticks(np.arange(-90, 30, 30))

    # Center-aligned text annotations with LaTeX rendering
    bbox_props = dict(boxstyle="square,pad=0.3", fc="white", ec="black", lw=1)
    
    ax.text(45, -45, "$C_L^* = 1.0$", ha="center", va="center", bbox=bbox_props, fontsize=12)
    ax.text(135, -30, "$C_L^* = -0.5$", ha="center", va="center", bbox=bbox_props, fontsize=12)
    
    ax.text(183, -45, f"V/V$_s$ = {v_slice}", va="center", fontsize=14)

    plt.tight_layout()
    
    output_path = Path(f"img/{prefix}_paper_policy_slice_V_{v_slice}.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    logger.info(f"[*] Plot successfully saved to {output_path.resolve()}")

# =====================================================================
# PIPELINE EXECUTION
# =====================================================================

def run_pipeline(setup_func: Any, prefix: str) -> None:
    """Execute training and analytics with injected configurations."""
    env, states, actions, config = setup_func()
    pi = train_or_load_policy(env, states, actions, config, prefix)
    
    if pi.n_dims == 3:
        plot_paper_style_policy_slice(pi, prefix, v_slice=1.2)

if __name__ == "__main__":
    run_pipeline(setup_banked_glider_experiment, "banked_glider")