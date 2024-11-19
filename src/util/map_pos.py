import os
import matplotlib.pyplot as plt

def main():
    # List and sort the log files in the "logs" directory
    logs = os.listdir("logs")
    logs = sorted(logs)
    
    print(f"Opening file : {logs[-1]}")

    # Read and parse the data from the latest log file
    with open(f"logs/{logs[-1]}", "r") as f:
        data = f.read().splitlines()

        # Split each line into components and convert them to floats
        data = [list(map(float, line.split())) for line in data]

    # Unpack the data into time, X, Y, and Z
    time, X, Y, Z = zip(*data)

    # Create a figure with multiple subplots
    fig, (ax_x, ax_y, ax_z) = plt.subplots(3, 1, figsize=(10, 8))

    # Scatter and line plots for each axis
    # ax_x.scatter(time, X, c='r', label='X Position')
    ax_x.plot(time, X, c='r', alpha=0.5)  # Line plot connecting the points
    ax_x.set_title("X vs Time")
    ax_x.set_xlabel("Time")
    ax_x.set_ylabel("X")

    # ax_y.scatter(time, Y, c='g', label='Y Position')
    ax_y.plot(time, Y, c='g', alpha=0.5)  # Line plot connecting the points
    ax_y.set_title("Y vs Time")
    ax_y.set_xlabel("Time")
    ax_y.set_ylabel("Y")

    # ax_z.scatter(time, Z, c='b', label='Z Position')
    ax_z.plot(time, Z, c='b', alpha=0.5)  # Line plot connecting the points
    ax_z.set_title("Z vs Time")
    ax_z.set_xlabel("Time")
    ax_z.set_ylabel("Z")

    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()