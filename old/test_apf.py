from typing import List, Tuple
from vector import Vector3D

def apf(
    current_pos: Vector3D,
    target_pos: Vector3D,
    obstacles: List[Tuple[Vector3D, float]],
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    z_bounds: Tuple[float, float],
    attraction_coeff_base: float = 1.0,
    repulsion_coeff: float = 1.0,
    normalise_val: float = 10
) -> Tuple[Vector3D, Vector3D, Vector3D, Vector3D]:
    """
    Calculate forces for the artificial potential field (APF) algorithm.

    Args:
        current_pos (Vector3D): Current position of the drone.
        target_pos (Vector3D): Target position for the drone.
        obstacles (List[Tuple[Vector3D, float]]): List of obstacles (position, radius).
        x_bounds (Tuple[float, float]): X-axis boundaries (min, max).
        y_bounds (Tuple[float, float]): Y-axis boundaries (min, max).
        z_bounds (Tuple[float, float]): Z-axis boundaries (min, max).
        attraction_coeff_base (float): Base coefficient for attraction force.
        repulsion_coeff (float): Coefficient for repulsion force.
        normalise_val (float): Normalisation value for attraction force.

    Returns:
        Tuple[Vector3D, Vector3D, Vector3D, Vector3D]: Delta to target, attractive force,
                                                      repulsive force, and total force.
    """
    # Calculate the difference between target and current position
    delta = target_pos - current_pos
    distance_goal = delta.magnitude()

    # Attractive force
    if distance_goal == 0:
        attractive_force = Vector3D(0, 0, 0)
    else:
        attraction_coeff = attraction_coeff_base * (distance_goal / normalise_val)
        attractive_force = Vector3D(
            attraction_coeff * delta.x / distance_goal,
            attraction_coeff * delta.y / distance_goal,
            attraction_coeff * delta.z / distance_goal
        )

    # Repulsive force for obstacles
    repulsive_force = Vector3D(0, 0, 0)
    for obs, radius in obstacles:
        delta_obs = current_pos - obs
        distance_obs = delta_obs.magnitude() - radius

        if distance_obs > 0:
            repulsion_strength = repulsion_coeff / (distance_obs**2)
            repulsive_force += Vector3D(
                repulsion_strength * delta_obs.x / delta_obs.magnitude(),
                repulsion_strength * delta_obs.y / delta_obs.magnitude(),
                repulsion_strength * delta_obs.z / delta_obs.magnitude()
            )

    # Boundary repulsion forces
    for axis, bounds in zip(('x', 'y', 'z'), (x_bounds, y_bounds, z_bounds)):
        pos_value = getattr(current_pos, axis)
        lower_bound, upper_bound = bounds

        # Lower boundary
        distance_to_lower = pos_value - lower_bound
        if distance_to_lower < 0:  # Inside the boundary
            repulsion_strength = repulsion_coeff / (distance_to_lower**2)
            repulsive_force += Vector3D(
                *(repulsion_strength if axis == 'x' else 0,
                  repulsion_strength if axis == 'y' else 0,
                  repulsion_strength if axis == 'z' else 0)
            )

        # Upper boundary
        distance_to_upper = upper_bound - pos_value
        if distance_to_upper < 0:  # Inside the boundary
            repulsion_strength = repulsion_coeff / (distance_to_upper**2)
            repulsive_force -= Vector3D(
                *(repulsion_strength if axis == 'x' else 0,
                  repulsion_strength if axis == 'y' else 0,
                  repulsion_strength if axis == 'z' else 0)
            )

    # Total force calculation: sum of attractive and repulsive forces
    total_force = attractive_force + repulsive_force

    return attractive_force, repulsive_force, total_force