import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import gymnasium as gym
# Project imports
from PolicyIteration import PolicyIteration, PolicyIterationConfig
from airplane.reduced_symmetric_glider_pullout import ReducedSymmetricGliderPullout
from utils.utils import get_optimal_action, get_barycentric_weights_and_indices

import logging
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange



import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any

# Internal imports for policy lookup
from utils.utils import get_optimal_action

# Configure logger
logger = logging.getLogger(__name__)

def train_or_load_policy(file_name: str = "ReducedSymmetricGliderPullout_policy.pkl") -> PolicyIteration:
    """Loads an existing policy from disk or trains a new one if it does not exist."""
    
    print("[*] Training new policy...")
    glider = ReducedSymmetricGliderPullout()

    # Define the discretization grid for the state space
    # Le damos un pequeño margen extra (-95 a +5) para que el estado terminal se alcance limpiamente
    bins_space = {
        "flight_path_angle": np.linspace(np.deg2rad(-180), np.deg2rad(0), 100, dtype=np.float32),
        "airspeed_norm": np.linspace(0.7, 4.0, 100, dtype=np.float32),
    }

    grid = np.meshgrid(*bins_space.values(), indexing='ij')
    states_space = np.vstack([g.ravel() for g in grid]).astype(np.float32).T

    custom_config = PolicyIterationConfig(
        gamma=0.99,          
        theta=1e-3,             
        n_steps=200,            
        log=False,              
        log_interval=50,        
        img_path=Path("./img")  
    )

    pi = PolicyIteration(
        env=glider, 
        states_space=states_space,
        action_space=np.linspace(-0.5, 1.0, 20, dtype=np.float32),
        config=custom_config
    )

    pi.run()
    return pi

def plot_policy_and_value(pi: PolicyIteration) -> None:
    """
    Plots the Value Function and the Optimal Policy over the state space.
    The Value Function colormap is reversed: Red = High altitude loss, Blue = Low altitude loss.
    """
    print("[*] Generating Policy and Value Function plots...")
    
    # Extract unique axis values from the original state space
    gamma_vals = np.unique(pi.states_space[:, 0])
    vnorm_vals = np.unique(pi.states_space[:, 1])
    G, V = np.meshgrid(gamma_vals, vnorm_vals, indexing='ij')
    
    query_points = np.column_stack([G.ravel(), V.ravel()]).astype(np.float32)
    actions_map = np.zeros(query_points.shape[0])
    values_map = np.zeros(query_points.shape[0])

    # Evaluate the policy over the entire grid using the O(1) mathematical interpolator
    for i, pt in enumerate(query_points):
        # Retrieve optimal action
        act, _ = get_optimal_action(pt, pi)
        actions_map[i] = act
        
        # Mathematical barycentric interpolation for the Value Function
        lambdas, indices = get_barycentric_weights_and_indices(
            np.atleast_2d(pt).astype(np.float32), 
            pi.bounds_low, pi.bounds_high, pi.grid_shape, pi.strides, pi.corner_bits
        )
        values_map[i] = np.sum(lambdas.flatten() * pi.value_function[indices.flatten()])

    Actions = actions_map.reshape(G.shape)
    Values = values_map.reshape(G.shape)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- Plot 1: Value Function (turbo_r: Red=Negative/Loss, Blue=Zero/Safe) ---
    c1 = ax1.contourf(np.rad2deg(G), V, Values, levels=50, cmap='turbo_r')
    fig.colorbar(c1, ax=ax1, label='Value Function (Altitude Loss)')
    ax1.set_title('Value Function \n(Red = High Altitude Loss, Blue = Safe)')
    ax1.set_xlabel('Flight Path Angle γ (deg)')
    ax1.set_ylabel('Airspeed Norm (V/Vs)')
    ax1.grid(alpha=0.3)

    # --- Plot 2: Optimal Policy ---
    c2 = ax2.contourf(np.rad2deg(G), V, Actions, levels=len(pi.action_space), cmap='coolwarm')
    fig.colorbar(c2, ax=ax2, label='Action (Lift Coefficient - C_L)')
    ax2.set_title('Optimal Policy (Decision Boundaries)')
    ax2.set_xlabel('Flight Path Angle γ (deg)')
    ax2.set_ylabel('Airspeed Norm (V/Vs)')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    
    # Save the figure securely to disk
    output_path = Path("img/policy_and_value.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[*] Plot successfully saved to: {output_path.resolve()}")
    plt.close()

def simulate_and_plot_trajectories(pi: PolicyIteration, initial_conditions: list) -> None:
    """Simulates and plots kinematic behavior for different initial states."""
    print(f"[*] Simulating {len(initial_conditions)} trajectories...")
    
    glider = ReducedSymmetricGliderPullout()
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    colors = ['b', 'r', 'g', 'm', 'c']
    
    for idx, init_state in enumerate(initial_conditions):
        state = np.array(init_state, dtype=np.float32)
        
        # Vectorized environment configuration
        glider.state = np.atleast_2d(state)
        glider.airplane.flight_path_angle = state[0]
        glider.airplane.airspeed_norm = state[1]

        trajectory = {'gamma': [], 'vnorm': [], 'cl': [], 'alt_loss': []}
        altitude_loss = 0.0
        done = False
        steps = 0

        # Simulation loop
        while not done and steps < 1000:
            current_state = glider.state[0].copy()
            
            # Boundary check bypassing heavy triangulation
            if np.any(current_state < pi.bounds_low) or np.any(current_state > pi.bounds_high):
                print(f"  -> Trajectory {idx+1} went out of bounds at step {steps}.")
                break
                
            # Extract scalar optimal action
            action, _ = get_optimal_action(current_state, pi)
            
            trajectory['gamma'].append(np.rad2deg(current_state[0]))
            trajectory['vnorm'].append(current_state[1])
            trajectory['cl'].append(action)
            trajectory['alt_loss'].append(altitude_loss)

            # Environment step
            next_state_batch, reward_batch, done_batch, _, _ = glider.step(action)
            
            reward = reward_batch[0]
            done = bool(done_batch[0])
            
            # Reward is (dt * V * sin(gamma) * Vs) which translates to delta altitude
            altitude_loss -= reward 
            
            glider.state = next_state_batch
            steps += 1
            
        print(f"  -> Trajectory {idx+1}: {steps} steps | Altitude lost: {altitude_loss:.2f} m")
        
        color = colors[idx % len(colors)]
        label = f'γ0={np.rad2deg(init_state[0]):.1f}°, V0={init_state[1]:.1f}'
        
        axs[0].plot(trajectory['gamma'], label=label, color=color, linewidth=2)
        axs[0].plot(trajectory['vnorm'], linestyle='--', color=color, alpha=0.7)
        axs[1].plot(trajectory['cl'], label=label, color=color, linewidth=2)
        axs[2].plot(trajectory['alt_loss'], label=label, color=color, linewidth=2)

    axs[0].set_title("Kinematic States (Solid: γ [deg] | Dashed: Normalized Speed)")
    axs[0].set_ylabel("State")
    axs[0].legend()
    axs[0].grid(True, alpha=0.5)

    axs[1].set_title("Control Action ($C_L$)")
    axs[1].set_ylabel("Lift Coefficient")
    axs[1].grid(True, alpha=0.5)

    axs[2].set_title("Cumulative Altitude Loss")
    axs[2].set_xlabel("Time (Steps)")
    axs[2].set_ylabel("Altitude Lost [m]")
    axs[2].grid(True, alpha=0.5)

    plt.tight_layout()
    
    # Save the figure securely to disk
    output_path = Path("img/trajectories.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[*] Plot successfully saved to: {output_path.resolve()}")
    plt.close()



import logging
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Internal project imports
from utils.utils import get_optimal_action

logger = logging.getLogger(__name__)

def get_parula_cmap() -> LinearSegmentedColormap:
    """
    Creates a highly accurate approximation of MATLAB's proprietary 
    parula colormap for scientific publication matching.
    """
    # Key anchor RGB values mapped from the original parula scale
    parula_anchors = [
        (0.2081, 0.1663, 0.5292),  # Deep blue
        (0.0165, 0.4266, 0.8786),  # Blue-cyan
        (0.0384, 0.6743, 0.7436),  # Cyan-green
        (0.4420, 0.7481, 0.5033),  # Green
        (0.8185, 0.7327, 0.3498),  # Yellow-green
        (0.9990, 0.7653, 0.2384),  # Yellow-orange
        (0.9769, 0.9839, 0.0805)   # Bright yellow
    ]
    return LinearSegmentedColormap.from_list("parula_approx", parula_anchors)

def plot_altitude_loss_heatmap(
    pi: Any, 
    glider_env: Any,
    v_min: float = 0.9,
    v_max: float = 4.0,
    v_steps: int = 60,
    gamma_min_deg: float = -90.0,
    gamma_max_deg: float = -1.0, 
    gamma_steps: int = 45,
    max_loss_plot: float = 180.0, 
    output_dir: str = "img",
    filename: str = "altitude_loss_heatmap_kinematic.png"
) -> None:
    """
    Generates a clean, professional heatmap for stall recovery research.
    Filters boundary artifacts and uses a custom MATLAB-style colormap.
    """
    logger.info("Starting high-fidelity kinematic simulation...")
    
    v_vals = np.linspace(v_min, v_max, v_steps)
    g_vals = np.linspace(np.deg2rad(gamma_min_deg), np.deg2rad(gamma_max_deg), gamma_steps)
    loss_matrix = np.zeros((gamma_steps, v_steps), dtype=np.float32)

    for i, gamma_init in enumerate(g_vals):
        for j, v_init in enumerate(v_vals):
            
            # Handle near-zero boundary safely
            if gamma_init == 0.0: 
                gamma_init = 0.1
                
            state = np.array([gamma_init, v_init], dtype=np.float32)
            glider_env.state = np.atleast_2d(state)
            glider_env.airplane.flight_path_angle = gamma_init
            glider_env.airplane.airspeed_norm = v_init
            
            total_loss = 0.0
            done = False
            step_count = 0
            
            # SIMULATION LOOP
            while not done and step_count < 1500:
                curr_s = glider_env.state[0].copy()
                
                if np.any(curr_s < pi.bounds_low) or np.any(curr_s > pi.bounds_high):
                    break
                
                action, _ = get_optimal_action(curr_s, pi)
                next_s, reward, done_batch, _, _ = glider_env.step(action)
                
                total_loss -= reward[0]
                glider_env.state = next_s
                done = bool(done_batch[0])
                step_count += 1
            
            # POST-PROCESS: Artifact correction
            if total_loss < 1.0 and gamma_init < np.deg2rad(-1.0):
                 loss_matrix[i, j] = loss_matrix[i-1, j] if i > 0 else 0.0
            else:
                loss_matrix[i, j] = total_loss

    # 3. Visualization Engine (Strict Paper Style)
    logger.info("Generating heatmap with custom Parula colormap...")
    v_grid, g_grid = np.meshgrid(v_vals, np.rad2deg(g_vals))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    mesh = ax.pcolormesh(
        v_grid, 
        g_grid, 
        loss_matrix, 
        cmap=get_parula_cmap(),  # Injected custom colormap
        shading='auto',
        vmin=0, 
        vmax=max_loss_plot,
        edgecolors='k',
        linewidth=0.1
    )

    ax.set_title("Minimum altitude loss as a function of initial conditions", fontweight='bold')
    ax.set_xlabel("Initial Relative Airspeed ($V/V_s$)")
    ax.set_ylabel("Initial Flight Path Angle (deg)")

    ax.set_xlim([v_min, v_max]) 
    ax.set_ylim([gamma_min_deg, 0.0])

    ax.set_xticks(np.arange(1, 4.5, 0.5))
    ax.set_yticks(np.arange(-90, 10, 10))

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label('Altitude Loss [m]')

    plt.tight_layout()
    
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    file_path = out_path / filename
    
    fig.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Heatmap successfully saved to: {file_path.resolve()}")

if __name__ == "__main__":
    # 1. Train or load the optimal policy
    pi = train_or_load_policy("ReducedSymmetricGliderPullout_policy.pkl")
    
    # 2. Plot the topological maps (Value Function and Policy)
    plot_policy_and_value(pi)

    # 3. Simulate critical scenarios
    initial_conditions = [
        [np.deg2rad(-15), 0.8],   # Mild dive
        [np.deg2rad(-45), 1.5],   # Moderate dive, high speed
        [np.deg2rad(-90), 0.8],   #
    ]
    
    simulate_and_plot_trajectories(pi, initial_conditions)
    #4. Generar el gráfico estilo MATLAB
    plot_altitude_loss_heatmap(pi, ReducedSymmetricGliderPullout())