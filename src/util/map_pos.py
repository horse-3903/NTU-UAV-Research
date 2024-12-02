import json
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def draw_sphere(ax, center, radius, resolution=50, color='c', alpha=0.3):
    """Draw a sphere on a 3D axis."""
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
    ax.plot_surface(x, y, z, color=color, alpha=alpha)


def load_config(log_dir):
    """Load configuration file from the specified log directory."""
    log_config_dir = f"{log_dir}/log-config.json"
    try:
        with open(log_config_dir, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Configuration file not found at {log_config_dir}")
        return {}


def load_log_data(log_dir):
    """Load log data from the most recent log file."""

    log_file = f"{log_dir}/log-pos.log"
    print(f"Opening file: {log_file}")

    with open(log_file, "r") as f:
        data = f.read().splitlines()
        return [list(map(float, line.split()[2:])) for line in data]


def create_2d_plots(fig, gs, time, X, Y, Z):
    """Create 2D plots of X, Y, Z positions against time."""
    ax_x = fig.add_subplot(gs[0, 0])
    ax_x.scatter(time, X, c='r', alpha=0.5)
    ax_x.plot(time, X, c='r', alpha=0.8)
    ax_x.set_title("X vs Time")
    ax_x.set_xlabel("Time")
    ax_x.set_ylabel("X")

    ax_y = fig.add_subplot(gs[1, 0])
    ax_y.scatter(time, Y, c='g', alpha=0.5)
    ax_y.plot(time, Y, c='g', alpha=0.8)
    ax_y.set_title("Y vs Time")
    ax_y.set_xlabel("Time")
    ax_y.set_ylabel("Y")

    ax_z = fig.add_subplot(gs[2, 0])
    ax_z.scatter(time, Z, c='b', alpha=0.5)
    ax_z.plot(time, Z, c='b', alpha=0.8)
    ax_z.set_title("Z vs Time")
    ax_z.set_xlabel("Time")
    ax_z.set_ylabel("Z")

    # Adjust vertical spacing between the 2D plots
    plt.subplots_adjust(hspace=0.3)

def create_3d_plot(fig, gs, X, Y, Z, takeoff_pos, start_pos, end_pos, target_pos, obstacles, path):
    """Create a 3D plot with positions, obstacles, and a takeoff plane."""
    ax_3d = fig.add_subplot(gs[:, 1], projection='3d')  # Place the 3D plot in the second column
    ax_3d.plot(X, Y, Z, c='purple', alpha=0.6, marker='o', label="Path")

    # Plot start, end, and target positions
    ax_3d.scatter(*takeoff_pos, color="yellow", label="Takeoff Position", s=100)
    ax_3d.scatter(*start_pos, color="red", label="Start Position", s=100)
    ax_3d.scatter(*end_pos, color="blue", label="End Position", s=100)
    ax_3d.scatter(*target_pos, color="green", label="Target Position", s=100)

    # Plot obstacles as spheres
    for obstacle in obstacles:
        center = obstacle[0]
        radius = obstacle[1]
        draw_sphere(ax_3d, center=center, radius=radius, color="grey", alpha=0.2)
        
    # Plot waypoints as spheres
    if path:
        for waypoint in path:
            draw_sphere(ax_3d, center=waypoint, radius=0.05, color="red", alpha=0.3)
        
    # Calculate midpoints
    max_range = max(max(X) - min(X), max(Y) - min(Y), max(Z) - min(Z))
    mid_x = (max(X) + min(X)) / 2
    mid_y = (max(Y) + min(Y)) / 2
    mid_z = (max(Z) + min(Z)) / 2

    # Set equal scaling for all axes
    ax_3d.set_box_aspect([1, 1, 1])
    ax_3d.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax_3d.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    ax_3d.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

    ax_3d.set_title("3D Scatter Plot of Position")
    ax_3d.set_xlabel("X")
    ax_3d.set_ylabel("Y")
    ax_3d.set_zlabel("Z")
    ax_3d.legend()


def main():
    log_dir = "logs"
    
    logs = sorted(os.listdir(log_dir))

    folder = logs[-1]
    
    log_dir = f"logs/{folder}"

    # Load configuration and log data
    config = load_config(log_dir)
    data = load_log_data(log_dir)

    # Unpack log data
    time, X, Y, Z = zip(*data)

    # Extract configuration values
    takeoff_pos = config.get("takeoff_pos", (0, 0, 0))
    start_pos = config.get("start_pos", (0, 0, 0))
    end_pos = config.get("end_pos", (0, 0, 0))
    target_pos = config.get("target_pos", (0, 0, 0))
    obstacles = config.get("obstacles", [])
    path = config.get("path", [])

    # Create a figure with a custom layout
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(3, 2, width_ratios=[1, 2])

    # Create 2D plots
    create_2d_plots(fig, gs, time, X, Y, Z)

    # Create 3D plot
    create_3d_plot(fig, gs, X, Y, Z, takeoff_pos, start_pos, end_pos, target_pos, obstacles, path)

    # Adjust layout and show plots
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()