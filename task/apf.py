from typing import List, Tuple
from vector import Vector3D

def apf(
    current_pos: Vector3D,
    target_pos: Vector3D,
    obstacles: List[Tuple[Vector3D, float]],
    attraction_coeff: float = 1.0,
    repulsion_coeff: float = 1.0,
    influence_distance: float = 5.0
) -> Tuple[Vector3D, Vector3D, Vector3D]:
    
    # Attractive force calculation
    delta = target_pos - current_pos
    distance_goal = delta.magnitude()

    if distance_goal == 0:
        attractive_force = Vector3D(0, 0, 0)
    else:
        attractive_force = Vector3D(
            attraction_coeff * distance_goal * delta.x / distance_goal,
            attraction_coeff * distance_goal * delta.y / distance_goal,
            attraction_coeff * distance_goal * delta.z / distance_goal
        )

    # Repulsive force calculation
    repulsive_force = Vector3D(0, 0, 0)
    for obs, radius in obstacles:
        delta_obs = current_pos - obs
        distance_obs = delta_obs.magnitude() - radius

        if 0 < distance_obs <= influence_distance:
            # Scale factor for the repulsive force
            repulsion_strength = repulsion_coeff * (1 / distance_obs - 1 / influence_distance) / (distance_obs**2)
            repulsive_force += Vector3D(
                repulsion_strength * delta_obs.x / delta_obs.magnitude(),
                repulsion_strength * delta_obs.y / delta_obs.magnitude(),
                repulsion_strength * delta_obs.z / delta_obs.magnitude()
            )

    # Total force calculation
    total_force = attractive_force + repulsive_force

    return total_force, attractive_force, repulsive_force