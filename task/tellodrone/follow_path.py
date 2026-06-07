"""APF-based path following task for the live drone.

The follow_path function is called on every UWB position update. It waits until
at least one depth estimation cycle has completed, then computes APF forces and
maps them to tellopy velocity commands.

Drone facing convention: the drone faces in the negative-X world direction, so:
  - forward()  → moves in -X
  - backward() → moves in +X
  - right()    → moves in +Y
  - left()     → moves in -Y
  - up()       → moves in +Z (or -Z depending on UWB frame)
  - down()     → moves in -Z
"""
import time

from apf import apf_with_bounds

from typing import TYPE_CHECKING, List, Tuple

from vector import Vector3D

if TYPE_CHECKING:
    from tellodrone.core import TelloDrone


def set_target_pos(self: "TelloDrone", target_pos: Vector3D) -> None:
    """Set the navigation goal position in world frame."""
    self.logger.info(f"Setting target position to {target_pos}")
    self.target_pos = target_pos


def add_obstacle(self: "TelloDrone", obstacle: Tuple[Vector3D, float]) -> None:
    """Manually add a known obstacle (centre, radius) to the obstacle list."""
    self.obstacles.append(obstacle)


def follow_path(self: "TelloDrone") -> None:
    """Navigate toward self.target_pos using APF, called each UWB callback.

    Blocks movement until the first depth estimation cycle has run.
    Triggers a new depth estimation on each call by setting the active video task.
    Shuts down cleanly when the drone is within 0.30 m of the target.
    """
    self.active_vid_task = self.run_depth_model
    if not self.depth_model_run:
        return

    local_delta = (self.cur_pos - self.target_pos).magnitude()

    if local_delta <= 0.30:
        self.logger.critical("Reached target position")
        self.shutdown(error=False, reason="Completed Follow Path Directive")
        return

    attract_coeff = 30
    repel_coeff = 10
    influence_dist = 0.5
    bounds_influence_dist = 0.5

    # Scale down attraction when close to avoid overshoot
    if local_delta < 1.0:
        attract_coeff = max(10, attract_coeff * local_delta)

    self.logger.debug(f"Current: {self.cur_pos}, Target: {self.target_pos}, Delta: {local_delta:.3f} m")

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

    max_val = 30
    scalar = 1

    velocity_x = round(total_force.x / local_delta * scalar)
    velocity_y = round(total_force.y / local_delta * scalar)
    velocity_z = round(total_force.z / local_delta * scalar)

    # Drone faces -X: positive world-X force → backward command
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

    self.logger.debug(f"Forces — total: {total_force}, attract: {attract_force}, repel: {repel_force}")
    self.logger.debug(f"Commands — X={velocity_x}, Y={velocity_y}, Z={velocity_z}")

    time.sleep(0.2)
