import matplotlib.pyplot as plt
import numpy as np

def draw_path(drone_pos: np.ndarray, target_pos: np.ndarray):
    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points
    ax.scatter(drone_pos[0], drone_pos[1], drone_pos[2], color='r', s=100, label='Drone Position')
    ax.scatter(target_pos[0], target_pos[1], target_pos[2], color='b', s=100, label='Target Position')

    # Draw the path (line) between the points
    x = [drone_pos[0], target_pos[0]]
    y = [drone_pos[1], target_pos[1]]
    z = [drone_pos[2], target_pos[2]]
    ax.plot(x, y, z, color='g', label='Path')

    # Set labels for the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Add a legend
    ax.legend()

    # Show the plot
    plt.show()
    
if __name__ == "__main__":
    drone_pos = np.array([1, 2, 3])  # Point 1 coordinates (x1, y1, z1)
    target_pos = np.array([4, 5, 6])  # Point 2 coordinates (x2, y2, z2)
    
    draw_path(drone_pos, target_pos)