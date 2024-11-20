import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

def main():
    # List and sort the log files in the "logs" directory
    logs = os.listdir("logs")
    logs = sorted(logs)
    folder = logs[-1]
    files = os.listdir(f"logs/{folder}")
    files = sorted(files)
    
    print(f"Opening file : logs/{folder}/{files[-1]}")

    # Read and parse the data from the latest log file
    with open(f"logs/{folder}/{files[-1]}", "r") as f:
        data = f.read().splitlines()

        # Split each line into components and convert them to floats
        data = [list(map(float, line.split()[2:])) for line in data]

    # Unpack the data into time, X, Y, and Z
    time, X, Y, Z = zip(*data)

    # Create a figure with a custom grid layout using gridspec
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(3, 2, width_ratios=[1, 2])  # Allocate more space to the second column

    # Create the three individual line plots on the left column
    ax_x = fig.add_subplot(gs[0, 0])
    ax_x.plot(time, X, c='r', alpha=0.5)
    ax_x.set_title("X vs Time")
    ax_x.set_xlabel("Time")
    ax_x.set_ylabel("X")

    ax_y = fig.add_subplot(gs[1, 0])
    ax_y.plot(time, Y, c='g', alpha=0.5)
    ax_y.set_title("Y vs Time")
    ax_y.set_xlabel("Time")
    ax_y.set_ylabel("Y")

    ax_z = fig.add_subplot(gs[2, 0])
    ax_z.plot(time, Z, c='b', alpha=0.5)
    ax_z.set_title("Z vs Time")
    ax_z.set_xlabel("Time")
    ax_z.set_ylabel("Z")

    # Create the 3D scatter plot on the right column, using more space
    ax_3d = fig.add_subplot(gs[:, 1], projection='3d')
    ax_3d.plot(X, Y, Z, c='purple', alpha=0.6, marker='o')
    ax_3d.set_title("3D Scatter Plot of Position")
    ax_3d.set_xlabel("X")
    ax_3d.set_ylabel("Y")
    ax_3d.set_zlabel("Z")

    # Adjust layout and show all plots
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()