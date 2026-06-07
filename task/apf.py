"""Artificial Potential Field (APF) controller for 3D drone navigation.

Implements attractive forces toward a target and repulsive forces from
obstacles and boundary walls. All positions and forces are in the world
coordinate frame used by the UWB localisation system.
"""
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
) -> Tuple[Vector3D, Vector3D, Vector3D]:
    """Compute APF forces without boundary walls.

    Args:
        cur_pos: Current drone position in world frame.
        target_pos: Goal position in world frame.
        obstacles: List of (centre, radius) obstacle spheres in world frame.
        attract_coeff: Attractive force gain.
        repel_coeff: Repulsive force gain.
        influence_dist: Distance (metres) at which obstacle repulsion activates.

    Returns:
        Tuple of (total_force, attractive_force, repulsive_force) as Vector3D.
    """
    direction_to_goal = target_pos - cur_pos
    distance_to_goal = direction_to_goal.magnitude()
    attractive_force = direction_to_goal.normalize() * (attract_coeff * distance_to_goal)

    repulsive_force = Vector3D(0, 0, 0)
    for obstacle_pos, obstacle_radius in obstacles:
        direction_to_obstacle = cur_pos - obstacle_pos
        distance_to_obstacle = direction_to_obstacle.magnitude() - obstacle_radius

        if distance_to_obstacle < influence_dist:
            repulsion_magnitude = repel_coeff * (
                (1.0 / distance_to_obstacle) - (1.0 / influence_dist)
            ) * (1.0 / (distance_to_obstacle ** 2))
            repulsive_force += direction_to_obstacle.normalize() * repulsion_magnitude

    total_force = attractive_force + repulsive_force
    return total_force, attractive_force, repulsive_force


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
) -> Tuple[Vector3D, Vector3D, Vector3D]:
    """Compute APF forces including repulsion from axis-aligned boundary walls.

    Boundary repulsion uses the same potential formula as obstacle repulsion,
    applied independently to each axis. This prevents the drone from drifting
    outside the configured flight envelope.

    Args:
        cur_pos: Current drone position in world frame.
        target_pos: Goal position in world frame.
        obstacles: List of (centre, radius) obstacle spheres in world frame.
        attract_coeff: Attractive force gain.
        repel_coeff: Repulsive force gain.
        influence_dist: Distance (metres) at which obstacle repulsion activates.
        x_bounds: (min, max) flight envelope along X axis.
        y_bounds: (min, max) flight envelope along Y axis.
        z_bounds: (min, max) flight envelope along Z axis.
        bounds_influence_dist: Distance from a wall at which boundary repulsion activates.

    Returns:
        Tuple of (total_force, attractive_force, repulsive_force) as Vector3D.
    """
    direction_to_target = target_pos - cur_pos
    distance_to_target = direction_to_target.magnitude()
    attractive_force = direction_to_target.normalize() * (attract_coeff * distance_to_target)

    repulsive_force = Vector3D(0, 0, 0)

    for obstacle_pos, obstacle_radius in obstacles:
        direction_to_obstacle = cur_pos - obstacle_pos
        distance_to_obstacle = direction_to_obstacle.magnitude() - obstacle_radius

        if distance_to_obstacle < influence_dist:
            repulsion_magnitude = repel_coeff * (
                (1.0 / distance_to_obstacle) - (1.0 / influence_dist)
            ) * (1.0 / (distance_to_obstacle ** 2))
            repulsive_force += direction_to_obstacle.normalize() * repulsion_magnitude

    def _apply_bound_repulsion(coord: float, min_bound: float, max_bound: float, axis: str) -> None:
        nonlocal repulsive_force
        if coord < min_bound + bounds_influence_dist:
            d = min_bound + bounds_influence_dist - coord
            if d < bounds_influence_dist:
                mag = repel_coeff * ((1.0 / d) - (1.0 / bounds_influence_dist)) * (1.0 / (d ** 2))
                if axis == "x":
                    repulsive_force += Vector3D(mag, 0, 0)
                elif axis == "y":
                    repulsive_force += Vector3D(0, mag, 0)
                elif axis == "z":
                    repulsive_force += Vector3D(0, 0, mag)
        elif coord > max_bound - bounds_influence_dist:
            d = coord - (max_bound - bounds_influence_dist)
            if d < bounds_influence_dist:
                mag = repel_coeff * ((1.0 / d) - (1.0 / bounds_influence_dist)) * (1.0 / (d ** 2))
                if axis == "x":
                    repulsive_force += Vector3D(-mag, 0, 0)
                elif axis == "y":
                    repulsive_force += Vector3D(0, -mag, 0)
                elif axis == "z":
                    repulsive_force += Vector3D(0, 0, -mag)

    _apply_bound_repulsion(cur_pos.x, x_bounds[0], x_bounds[1], "x")
    _apply_bound_repulsion(cur_pos.y, y_bounds[0], y_bounds[1], "y")
    _apply_bound_repulsion(cur_pos.z, z_bounds[0], z_bounds[1], "z")

    total_force = attractive_force + repulsive_force
    return total_force, attractive_force, repulsive_force
