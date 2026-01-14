import os
import sys

import matplotlib.pyplot as plt
import numpy as np


def compare_nav_logs(sim_path, real_path):
    # Load Real (Homi) Log
    if not os.path.exists(real_path):
        print(f"Real log not found: {real_path}")
        real_dict = None
    else:
        print(f"Loading real log from {real_path}...")
        try:
            real_dict = np.load(real_path)
            print(f"Real keys: {list(real_dict.keys())}")
        except Exception as e:
            print(f"Error loading real log: {e}")
            real_dict = None

    # Load Sim Log
    if not os.path.exists(sim_path):
        print(f"Sim log not found: {sim_path}")
        sim_dict = None
    else:
        print(f"Loading sim log from {sim_path}...")
        try:
            sim_dict = np.load(sim_path)
            print(f"Sim keys: {list(sim_dict.keys())}")
        except Exception as e:
            print(f"Error loading sim log: {e}")
            sim_dict = None

    if real_dict is None and sim_dict is None:
        print("No data to plot.")
        return

    # Use keys from sim or real dict to determine plots
    all_keys = (
        sorted(list(set(sim_dict.keys()) | set(real_dict.keys())))
        if (sim_dict is not None and real_dict is not None)
        else list(sim_dict.keys()) if sim_dict is not None else list(real_dict.keys())
    )

    # Mapping for descriptive titles
    titles_map = {
        "base_lin_vel": ["Lin Vel X", "Lin Vel Y", "Lin Vel Z"],
        "base_ang_vel": ["Ang Vel X", "Ang Vel Y", "Ang Vel Z"],
        "euler_rpy": ["Roll", "Pitch", "Yaw"],
        "projected_gravity": ["Gravity X", "Gravity Y", "Gravity Z"],
        "nav_commands": ["Sigma X", "Sigma Y", "Sigma Z"],
        "task_flag": ["Task Flag"],
        "actions": ["Action Vx", "Action Vy", "Action Vyaw", "Action Pitch"],
    }

    plot_tasks = []
    for key in all_keys:
        # Get sample data to determine dimensions
        sample = None
        if sim_dict is not None and key in sim_dict:
            sample = sim_dict[key]
        elif real_dict is not None and key in real_dict:
            sample = real_dict[key]

        if sample is None:
            continue

        # Ensure 2D
        if len(sample.shape) == 1:
            sample = sample[:, np.newaxis]

        n_dims = sample.shape[1]

        # Logic for determining number of dimensions to plot
        plot_dims = n_dims
        if key == "nav_commands" and n_dims > 3:
            plot_dims = 3  # Only plot first 3 cmds by default

        for d in range(plot_dims):
            title = ""
            if key in titles_map and d < len(titles_map[key]):
                title = titles_map[key][d]
            else:
                # Extensible: format the key name nicely
                clean_key = key.replace("_", " ").title()
                title = f"{clean_key} {d}" if n_dims > 1 else clean_key

            plot_tasks.append((key, d, title))

    n_plots = len(plot_tasks)
    if n_plots == 0:
        print("No valid plot tasks identified.")
        return

    # Filter unique titles to avoid duplicates if key 'data' and separate keys exist
    # (Though in our case it's usually one or the other)

    # Setup plot grid
    cols = 4
    rows = (n_plots + cols - 1) // cols

    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    fig.suptitle(f"Isaac vs MuJoCo Nav Agent Log Analysis", fontsize=16)

    if n_plots == 1:
        ax_flat = [axs]
    else:
        ax_flat = axs.flatten()

    # Time axis assumption: 10Hz (0.1s step)

    for i in range(len(ax_flat)):
        ax = ax_flat[i]
        if i < n_plots:
            key, dim, title = plot_tasks[i]

            # Plot Real (MuJoCo) - RED
            if real_dict is not None and key in real_dict:
                data = real_dict[key]
                if len(data.shape) == 1:
                    data = data[:, np.newaxis]
                if dim < data.shape[1]:
                    t = np.arange(len(data)) * 0.1
                    ax.plot(t, data[:, dim], label="MuJoCo", color="red", alpha=0.8, linewidth=1.5)

            # Plot Sim (Isaac) - BLUE
            if sim_dict is not None and key in sim_dict:
                data = sim_dict[key]
                if len(data.shape) == 1:
                    data = data[:, np.newaxis]
                if dim < data.shape[1]:
                    t = np.arange(len(data)) * 0.1
                    ax.plot(t, data[:, dim], label="Isaac", color="blue", alpha=0.6, linestyle="--")

            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("Time (s)")
            ax.grid(True, which="both", linestyle="--", alpha=0.5)
            if i == 0:
                ax.legend(loc="upper right")
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.93, hspace=0.5)
    print("Showing comparison plots...")
    plt.show()


if __name__ == "__main__":
    default_sim = os.path.expanduser("~/sim_nav_log.npz")
    default_real = os.path.expanduser("~/homi_nav_log.npz")

    sim_path = sys.argv[1] if len(sys.argv) > 1 else default_sim
    real_path = sys.argv[2] if len(sys.argv) > 2 else default_real

    compare_nav_logs(sim_path, real_path)
