from typing import List, Tuple

from vector import Vector3D

def apf(
    current_pos: Vector3D,
    target_pos: Vector3D,
    obstacles: List[Tuple[Vector3D, float]],
    attraction_coeff: float = 1.0,
    repulsion_coeff: float = 1.0,
    normalise_val: float = 10,
) -> Tuple[Vector3D, Vector3D, Vector3D]:
    # Calculate the difference between target and current position
    delta = target_pos - current_pos
    distance_goal = delta.magnitude()

    # Attractive force
    if distance_goal == 0:
        attractive_force = Vector3D(0.0, 0.0, 0.0)
    else:
        attraction_coeff = attraction_coeff * (distance_goal / normalise_val)
        attractive_force = delta.normalize() * attraction_coeff

    # Repulsive force calculation
    repulsive_force = Vector3D(0.0, 0.0, 0.0)
    
    for obs, radius in obstacles:
        delta_obs = current_pos - obs
        distance_obs = delta_obs.magnitude() - radius

        if distance_obs > 0:
            repulsion_coeff = repulsion_coeff / (distance_obs**2)
                
            repulsive_component = delta_obs.normalize() * repulsion_coeff
            repulsive_force += repulsive_component

    # Total force
    total_force = attractive_force + repulsive_force

    return attractive_force, repulsive_force, total_force