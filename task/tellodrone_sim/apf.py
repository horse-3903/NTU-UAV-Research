import math
from vector import Vector3D
from typing import List, Tuple

def apf(
    cur_pos: Vector3D,
    target_pos: Vector3D,
    obstacles: List[Tuple[Vector3D, float]],
    attract_coeff: float,
    repel_coeff: float,
    influence_dist: float,
):
    """
    Artificial Potential Fields with 3D vector support.

    Parameters:
        cur_pos (Vector3D): Current position [x, y, z].
        target_pos (Vector3D): Goal position [x, y, z].
        obstacles (list[tuple[Vector3D, float]]): List of obstacles as (position, radius).
        attract_coeff (float): Gain coefficient for the attractive field.
        repel_coeff (float): Gain coefficient for the repulsive field.
        influence_dist (float): Distance threshold for repulsion to be active.

    Returns:
        Vector3D: Velocity direction for the next step [x, y, z].
        float: Heading angle in radians.
        float: Magnitude of the attractive potential field.
        float: Magnitude of the repulsive potential field.
    """
    # Calculate the attractive force
    direction_to_goal = target_pos - cur_pos
    distance_to_goal = direction_to_goal.magnitude()
    attractive_force = direction_to_goal.normalize() * (attract_coeff * distance_to_goal)
    attractive_potential = 0.5 * attract_coeff * (distance_to_goal**2)

    # Calculate the repulsive force
    total_repulsive_force = Vector3D(0, 0, 0)
    repulsive_potential = 0

    for obstacle_pos, obstacle_radius in obstacles:
        direction_to_obstacle = cur_pos - obstacle_pos
        distance_to_obstacle = direction_to_obstacle.magnitude() - obstacle_radius

        if distance_to_obstacle < influence_dist:
            repulsion_magnitude = repel_coeff * (
                (1.0 / distance_to_obstacle) - (1.0 / influence_dist)
            ) * (1.0 / (distance_to_obstacle**2))
            repulsive_force = direction_to_obstacle.normalize() * repulsion_magnitude
            total_repulsive_force += repulsive_force
            repulsive_potential += 0.5 * repel_coeff * (
                (1.0 / distance_to_obstacle) - (1.0 / influence_dist)
            )**2

    # Calculate the total force
    total_force = attractive_force + total_repulsive_force

    # Calculate heading angle
    heading_angle = math.atan2(total_force.y, total_force.x)

    # Return velocity direction, heading angle, and potential values
    return total_force, heading_angle, attractive_potential, repulsive_potential

def apf_with_bounds(
    cur_pos: Vector3D,
    target_pos: Vector3D,
    obstacles: List[Tuple[Vector3D, float]],
    attract_coeff: float,
    repel_coeff: float,
    influence_dist: float,
    x_bounds: Tuple[float, float],
    y_bounds: Tuple[float, float],
    z_bounds: Tuple[float, float],
    bounds_influence_dist: float,
):
    """
    Artificial Potential Fields with 3D vector support and boundary repulsion.

    Parameters:
        cur_pos (Vector3D): Current position [x, y, z].
        target_pos (Vector3D): Goal position [x, y, z].
        obstacles (list[tuple[Vector3D, float]]): List of obstacles as (position, radius).
        attract_coeff (float): Gain coefficient for the attractive field.
        repel_coeff (float): Gain coefficient for the repulsive field.
        influence_dist (float): Distance threshold for repulsion to be active.
        x_bounds (tuple[float, float]): X-axis bounds (min, max).
        y_bounds (tuple[float, float]): Y-axis bounds (min, max).
        z_bounds (tuple[float, float]): Z-axis bounds (min, max).
        bounds_influence_dist (float): Distance threshold for bounds repulsion to be active.

    Returns:
        Vector3D: Velocity direction for the next step [x, y, z].
        float: Heading angle in radians.
        float: Magnitude of the attractive potential field.
        float: Magnitude of the repulsive potential field.
    """
    # Calculate the attractive force
    direction_to_goal = target_pos - cur_pos
    distance_to_goal = direction_to_goal.magnitude()
    attractive_force = direction_to_goal.normalize() * (attract_coeff * distance_to_goal)
    attractive_potential = 0.5 * attract_coeff * (distance_to_goal**2)

    # Calculate the repulsive force
    total_repulsive_force = Vector3D(0, 0, 0)
    repulsive_potential = 0

    # Repulsion from obstacles
    for obstacle_pos, obstacle_radius in obstacles:
        direction_to_obstacle = cur_pos - obstacle_pos
        distance_to_obstacle = direction_to_obstacle.magnitude() - obstacle_radius

        if distance_to_obstacle < influence_dist:
            repulsion_magnitude = repel_coeff * (
                (1.0 / distance_to_obstacle) - (1.0 / influence_dist)
            ) * (1.0 / (distance_to_obstacle**2))
            repulsive_force = direction_to_obstacle.normalize() * repulsion_magnitude
            total_repulsive_force += repulsive_force
            repulsive_potential += 0.5 * repel_coeff * (
                (1.0 / distance_to_obstacle) - (1.0 / influence_dist)
            )**2

    # Repulsion from bounds
    bounds = [
        (x_bounds[0], x_bounds[1], cur_pos.x, Vector3D(1, 0, 0)),
        (y_bounds[0], y_bounds[1], cur_pos.y, Vector3D(0, 1, 0)),
        (z_bounds[0], z_bounds[1], cur_pos.z, Vector3D(0, 0, 1)),
    ]

    for lower, upper, coord, axis_vector in bounds:
        for boundary in [(lower, -axis_vector), (upper, axis_vector)]:
            bound_pos, bound_direction = boundary
            distance_to_bound = abs(coord - bound_pos)

            if distance_to_bound < bounds_influence_dist:
                repulsion_magnitude = repel_coeff * (
                    (1.0 / distance_to_bound) - (1.0 / bounds_influence_dist)
                ) * (1.0 / (distance_to_bound**2))
                repulsive_force = bound_direction * repulsion_magnitude
                total_repulsive_force += repulsive_force
                repulsive_potential += 0.5 * repel_coeff * (
                    (1.0 / distance_to_bound) - (1.0 / bounds_influence_dist)
                )**2

    # Calculate the total force
    total_force = attractive_force + total_repulsive_force

    # Calculate heading angle
    heading_angle = math.atan2(total_force.y, total_force.x)

    # Return velocity direction, heading angle, and potential values
    return total_force, heading_angle, attractive_potential, repulsive_potential
