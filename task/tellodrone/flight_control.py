from typing import Tuple
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tellodrone.core import TelloDrone

def flight_data_callback(self: "TelloDrone", event, sender, data):
    data = str(data).split(" | ")
    data = [d.split(": ") for d in data]
    data = [(k.strip(), v.strip()) for k, v in data]
    data = {k: v for k, v in data}
    
    self.altitude = float(data["ALT"])
    self.speed = float(data["SPD"])
    self.battery = float(data["BAT"])
    self.wifi = float(data["WIFI"])
    self.cam = float(data["CAM"])
    self.mode = float(data["MODE"])
    
    if self.battery <= 5:
        self.logger.critical(f"Drone battery very low at {self.battery}%")
        self.shutdown(error=True, reason=f"Insufficient battery at {self.battery}%")
    elif self.battery <= 10:
        self.logger.warning(f"Drone battery low at {self.battery}%")

def check_bounds(self: "TelloDrone", x_bounds: Tuple[float, float], y_bounds: Tuple[float, float], z_bounds: Tuple[float, float]):
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    z_min, z_max = z_bounds
    
    if not (x_min <= self.cur_pos.x <= x_max):
        self.shutdown(error=True, reason=f"Out of X bounds : {self.cur_pos.x}")
        
    if not (y_min <= self.cur_pos.y <= y_max):
        self.shutdown(error=True, reason=f"Out of Y bounds : {self.cur_pos.y}")
        
    if not (z_min <= self.cur_pos.z <= z_max):
        self.shutdown(error=True, reason=f"Out of Z bounds : {self.cur_pos.z}")
