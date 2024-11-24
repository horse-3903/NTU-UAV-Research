from typing import List, Tuple
from vector import Vector3D

def apf(
    current_pos: Vector3D,
    target_pos: Vector3D,
    obstacles: List[Tuple[Vector3D, float]],
    attraction_coeff_base: float = 1.0,
    repulsion_coeff: float = 1.0,
    normalise_val: float = 10
) -> Tuple[Vector3D, Vector3D, Vector3D]:
    
    # Calculate the difference between target and current position
    delta = target_pos - current_pos
    distance_goal = delta.magnitude()

    if distance_goal == 0:
        attractive_force = Vector3D(0, 0, 0)
    else:
        attraction_coeff = attraction_coeff_base * (distance_goal / normalise_val)
        
        attractive_force = Vector3D(
            attraction_coeff * delta.x,
            attraction_coeff * delta.y,
            attraction_coeff * delta.z
        )

    # Repulsive force calculation (always applied)
    repulsive_force = Vector3D(0, 0, 0)
    
    for obs, radius in obstacles:
        delta_obs = current_pos - obs
        distance_obs = delta_obs.magnitude() - radius

        # Repulsive force strength is calculated for all obstacles
        if abs(distance_obs) > 0:
            repulsion_strength = repulsion_coeff / (distance_obs**2)
            repulsive_force += Vector3D(
                repulsion_strength * delta_obs.x / delta_obs.magnitude(),
                repulsion_strength * delta_obs.y / delta_obs.magnitude(),
                repulsion_strength * delta_obs.z / delta_obs.magnitude()
            )

    # Total force calculation: sum of attractive and repulsive forces
    total_force = attractive_force + repulsive_force

    return attractive_force, repulsive_force, total_force