import time

from apf import apf, apf_with_bounds

from typing import TYPE_CHECKING
from typing import List, Tuple

from vector import Vector3D

if TYPE_CHECKING:
    from tellodrone.core import TelloDrone

def set_target_pos(self: "TelloDrone", target_pos: Vector3D):
    self.logger.info(f"Setting target position to {self.target_pos}")
    self.target_pos = target_pos

    
def add_obstacle(self: "TelloDrone", obstacle: Tuple[Vector3D, float]):
    self.obstacles.append(obstacle)

    
def follow_path(self: "TelloDrone") -> None:
    self.active_vid_task = self.run_depth_model
    if not self.depth_model_run:
        return

    local_delta = (self.cur_pos - self.target_pos).magnitude()

    if local_delta <= 0.30:
        self.logger.critical(f"Reached target position")
        self.shutdown(error=False, reason="Completed Follow Path Directive")
        return 

    # Potential field parameters
    attract_coeff = 30
    repel_coeff = 10
    influence_dist = 0.5
    bounds_influence_dist = 0.5

    if local_delta < 1.0:
        attract_coeff = max(10  , attract_coeff * local_delta)

    self.logger.debug(f"Current Position: {self.cur_pos}")
    self.logger.debug(f"Target Position: {self.target_pos}")
    self.logger.debug(f"Local Delta: {local_delta}")

    # Calculate forces using APF
    total_force, attract_force, repel_force = apf_with_bounds(
        cur_pos=self.cur_pos,
        target_pos=self.target_pos,
        obstacles=self.obstacles,
        attract_coeff=attract_coeff,
        repel_coeff=repel_coeff,
        influence_dist=influence_dist,
        x_bounds=self.x_bounds,
        y_bounds=self.y_bounds,
        z_bounds=self.z_bounds,
        bounds_influence_dist=bounds_influence_dist,
    )

    scalar = 1
    max_val = 30

    force_x = total_force.x
    force_y = total_force.y
    force_z = total_force.z

    velocity_x = round(force_x / local_delta * scalar)
    velocity_y = round(force_y / local_delta * scalar)
    velocity_z = round(force_z / local_delta * scalar)

    # Drone movement commands (assumes facing negative-x)
    if velocity_x < 0:
        self.drone.forward(min(max_val, abs(velocity_x)))
    else:
        self.drone.backward(min(max_val, abs(velocity_x)))

    if velocity_y > 0:
        self.drone.right(min(max_val, abs(velocity_y)))
    else:
        self.drone.left(min(max_val, abs(velocity_y)))

    if velocity_z < 0:
        self.drone.down(min(max_val, abs(velocity_z)))
    else:
        self.drone.up(min(max_val, abs(velocity_z)))

    self.logger.debug(f"Resultant Force: {total_force}")
    self.logger.debug(f"Attractive Force: {attract_force}")
    self.logger.debug(f"Repulsive Force: {repel_force}")
    self.logger.debug(f"Control signals: X={velocity_x}, Y={velocity_y}, Z={velocity_z}")
    self.logger.debug(f"Current Position: {self.cur_pos}")
    self.logger.debug(f"Target Position: {self.target_pos}")
    
    time.sleep(0.2)