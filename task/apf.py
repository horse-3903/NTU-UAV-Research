import math
from vector import Vector3D
from typing import List, Tuple

def apf(cur_pos: Vector3D, target_pos: Vector3D, obstacles: List[Tuple[Vector3D, float]], attract_coeff: float, repel_coeff: float, influence_dist: float):
    # Calculate the attractive force
    direction_to_goal = target_pos - cur_pos
    distance_to_goal = direction_to_goal.magnitude()
    attractive_force = direction_to_goal.normalize() * (attract_coeff * distance_to_goal)
    attractive_potential = 0.5 * attract_coeff * (distance_to_goal**2)

    # Calculate the repulsive force
    repulsive_force = Vector3D(0, 0, 0)
    repulsive_potential = 0

    for obstacle_pos, obstacle_radius in obstacles:
        direction_to_obstacle = cur_pos - obstacle_pos
        distance_to_obstacle = direction_to_obstacle.magnitude() - obstacle_radius

        if distance_to_obstacle < influence_dist:
            repulsion_magnitude = repel_coeff * (
                (1.0 / distance_to_obstacle) - (1.0 / influence_dist)
            ) * (1.0 / (distance_to_obstacle**2))
            repulsive_force += direction_to_obstacle.normalize() * repulsion_magnitude
            repulsive_potential += 0.5 * repel_coeff * (
                (1.0 / distance_to_obstacle) - (1.0 / influence_dist)
            )**2

    # Calculate the total force
    total_force = attractive_force + repulsive_force

    # Calculate heading angle
    heading_angle = math.atan2(total_force.y, total_force.x)

    # Return velocity direction, heading angle, and potential values
    return (
        total_force,
        attractive_force,
        repulsive_force,
    )

def apf_with_bounds(cur_pos: Vector3D, target_pos: Vector3D, obstacles: List[Tuple[Vector3D, float]], attract_coeff: float, repel_coeff: float, influence_dist: float, x_bounds: Tuple[float, float], y_bounds: Tuple[float, float], z_bounds: Tuple[float, float], bounds_influence_dist: float):
    # Calculate the attractive force
    direction_to_target = target_pos - cur_pos
    distance_to_target = direction_to_target.magnitude()
    attractive_force = direction_to_target.normalize() * (attract_coeff * distance_to_target)
    
    # Initialize repulsive force and potential
    repulsive_force = Vector3D(0, 0, 0)

    # Calculate repulsive force from obstacles
    for obstacle_pos, obstacle_radius in obstacles:
        direction_to_obstacle = cur_pos - obstacle_pos
        distance_to_obstacle = direction_to_obstacle.magnitude() - obstacle_radius

        if distance_to_obstacle < influence_dist:
            repulsion_magnitude = repel_coeff * (
                (1.0 / distance_to_obstacle) - (1.0 / influence_dist)
            ) * (1.0 / (distance_to_obstacle**2))
            repulsive_force += direction_to_obstacle.normalize() * repulsion_magnitude

    # Calculate repulsive force from bounds
    def calculate_bound_repulsion(coord, min_bound, max_bound, axis):
        nonlocal repulsive_force
        if coord < min_bound + bounds_influence_dist:
            distance_to_bound = min_bound + bounds_influence_dist - coord
            if distance_to_bound < bounds_influence_dist:
                repulsion_magnitude = repel_coeff * (
                    (1.0 / distance_to_bound) - (1.0 / bounds_influence_dist)
                ) * (1.0 / (distance_to_bound**2))
                if axis == "x":
                    repulsive_force += Vector3D(repulsion_magnitude, 0, 0)
                elif axis == "y":
                    repulsive_force += Vector3D(0, repulsion_magnitude, 0)
                elif axis == "z":
                    repulsive_force += Vector3D(0, 0, repulsion_magnitude)
        elif coord > max_bound - bounds_influence_dist:
            distance_to_bound = coord - (max_bound - bounds_influence_dist)
            if distance_to_bound < bounds_influence_dist:
                repulsion_magnitude = repel_coeff * (
                    (1.0 / distance_to_bound) - (1.0 / bounds_influence_dist)
                ) * (1.0 / (distance_to_bound**2))
                if axis == "x":
                    repulsive_force += Vector3D(-repulsion_magnitude, 0, 0)
                elif axis == "y":
                    repulsive_force += Vector3D(0, -repulsion_magnitude, 0)
                elif axis == "z":
                    repulsive_force += Vector3D(0, 0, -repulsion_magnitude)

    calculate_bound_repulsion(cur_pos.x, x_bounds[0], x_bounds[1], "x")
    calculate_bound_repulsion(cur_pos.y, y_bounds[0], y_bounds[1], "y")
    calculate_bound_repulsion(cur_pos.z, z_bounds[0], z_bounds[1], "z")

    # Calculate the total force
    total_force = attractive_force + repulsive_force

    # Return total force, attractive force, and repulsive force
    return total_force, attractive_force, repulsive_force