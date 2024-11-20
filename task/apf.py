from vector import Vector3D
import numpy as np

def apf(current_pos: Vector3D, target_pos: Vector3D, attraction_coeff: float = 1.0) -> tuple:
    """
    Artificial Potential Fields for 3D, using Vector3D class.
    
    current_pos: Current position as a Vector3D
    target_pos: Target position as a Vector3D
    attraction_coeff: Attraction coefficient
    
    Returns:
    x_vel: Velocity in the x direction
    y_vel: Velocity in the y direction
    z_vel: Velocity in the z direction
    """
    # Calculate the difference vector
    delta = target_pos - current_pos
    
    # Calculate the attractive force angle in the XY plane
    distance_xy = np.sqrt(delta.x**2 + delta.y**2)
    if distance_xy > 0:
        angle_attract_xy = np.arctan2(delta.y, delta.x)
    else:
        angle_attract_xy = 0
    
    # Calculate the attractive force angle in the Z direction (vertical angle)
    angle_attract_z = np.arctan2(delta.z, distance_xy) if distance_xy != 0 else np.pi / 2
    
    # Calculate the attractive force magnitude
    distance_goal = delta.magnitude()
    force_attract = Vector3D(
        attraction_coeff * distance_goal * np.cos(angle_attract_z) * np.cos(angle_attract_xy),
        attraction_coeff * distance_goal * np.cos(angle_attract_z) * np.sin(angle_attract_xy),
        attraction_coeff * distance_goal * np.sin(angle_attract_z)
    )
    
    # Calculate the velocities based on the force direction
    x_vel = force_attract.x / distance_goal  # Normalize by the distance
    y_vel = force_attract.y / distance_goal
    z_vel = force_attract.z / distance_goal
    
    return x_vel, y_vel, z_vel