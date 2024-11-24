from datetime import datetime

from vector import Vector3D

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tellodrone.core import TelloDrone
    
def task_handler(self: "TelloDrone", pos_arr: list) -> None:
    self.cur_pos = Vector3D.from_arr(pos_arr)
    self.cur_time = datetime.now()
    
    delta_time = self.cur_time - self.init_time
    
    with open(self.log_pos_file, "a") as f:
        f.write(f"{self.cur_time} {delta_time.total_seconds()} {self.cur_pos.x} {self.cur_pos.y} {self.cur_pos.z}" + "\n")
    
    self.check_bounds(self.x_bounds, self.y_bounds, self.z_bounds)
    
    if self.active_task is not None:
        self.active_task()
    elif self.running:
        self.shutdown()
    
def run_objective(self: "TelloDrone") -> None:
        self.logger.info("Running objective")
        self.startup()
        
        self.active_task = self.follow_path
        
        if self.running and self.active_task is None:
            self.shutdown()