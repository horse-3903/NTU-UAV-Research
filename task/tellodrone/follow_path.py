import time

from task.apf import apf

from typing import TYPE_CHECKING
from typing import List, Tuple

from task.vector import Vector3D

if TYPE_CHECKING:
    from tellodrone.core import TelloDrone

def set_target_pos(self: "TelloDrone", target_pos: Vector3D):
    self.logger.info(f"Setting target position to {target_pos}")
    self.target_pos = target_pos
    
def set_obstacles(self: "TelloDrone", obstacles: List[Tuple[Vector3D, float]]):
    self.obstacles.extend(obstacles)
    
def follow_path(self: "TelloDrone") -> None:
    if not self.target_pos:
        self.logger.error("Path not planned. Call TelloDrone.plan_path() first.")
    
    self.active_vid_task = self.run_depth_model
    local_delta = (self.cur_pos - self.target_pos).magnitude()
    
    if local_delta <= 0.4:
        self.logger.info("Drone has reached target")
        self.active_vid_task = None
        self.active_task = None
    
    # to change
    attract_coeff = 80
    repul_coeff = 20
    
    global_delta = (self.start_pos - self.target_pos).magnitude()
    
    total_force, attract_force, repel_force = apf(
        current_pos=self.cur_pos, 
        target_pos=self.target_pos, 
        obstacles=self.obstacles, 
        # x_bounds=x_bounds,
        # y_bounds=y_bounds,
        # z_bounds=z_bounds,
        attraction_coeff=attract_coeff, 
        repulsion_coeff=repul_coeff, 
        normalise_val=global_delta)
    
    scalar = 1
    
    force_x = total_force.x
    force_y = total_force.y
    force_z = total_force.z
    
    velocity_x = round(force_x / local_delta * scalar)
    velocity_y = round(force_y / local_delta * scalar)
    velocity_z = round(force_z / local_delta * scalar)

    # self.logger control signals
    self.logger.debug(f"Resultant Force : {total_force}")
    self.logger.debug(f"Attractive Force : {attract_force}")
    self.logger.debug(f"Repulsive Force : {repel_force}")
    self.logger.debug(f"Control signals: X={velocity_x}, Y={velocity_y}, Z={velocity_z}")

    # assuming facing towards negative-x
    if velocity_x < 0:
        self.drone.forward(abs(velocity_x))
    else:
        self.drone.backward(abs(velocity_x))

    if velocity_y > 0:
        self.drone.right(abs(velocity_y))
    else:
        self.drone.left(abs(velocity_y))

    if velocity_z < 0:
        self.drone.down(abs(velocity_z))
    else:
        self.drone.up(abs(velocity_z))

    time.sleep(0.2)
    self.logger.debug(f"Current position : {self.cur_pos}")
    self.logger.debug(f"Target position : {self.target_pos}")