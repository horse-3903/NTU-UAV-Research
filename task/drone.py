import sys
import time

import av.container
import rospy

from tellopy import Tello

import math
import numpy as np
from vector import Vector3D

import av

from simple_pid import PID

class TelloDrone:
    def __init__(self) -> None:
        self.cur_pos = Vector3D(0, 0, 0)
        self.target_pos = Vector3D(0, 0, 0)
        self.orientation = None
        
        self.active_task = lambda : None
        
        self.drone = Tello()
        self.running = False
        
        self.waypts = []
        self.cur_waypt_idx = -1
        
        self.container = None
        
        self.pid_x = PID(Kp=1.0, Ki=0.1, Kd=0.05, output_limits=(-10, 10))
        self.pid_y = PID(Kp=1.0, Ki=0.1, Kd=0.05, output_limits=(-10, 10))
        self.pid_z = PID(Kp=1.0, Ki=0.1, Kd=0.05, output_limits=(-10, 10))
        
    def set_target(self, target: Vector3D) -> None:
        self.target = target
        
    def orient_drone(self) -> None:
        f_pos = self.cur_pos
        
        self.drone.forward(10)
        time.sleep(1)
        
        s_pos = self.cur_pos
        
        x_diff = f_pos.x - s_pos.x
        z_diff = f_pos.z - s_pos.z
        
        self.orientation = np.arctan2(z_diff, x_diff)
        self.orientation = np.degrees(self.orientation)
        
        if self.orientation < -180:
            self.orientation += 360
        elif self.orientation > 180:
            self.orientation -= 360
        
    def startup(self) -> None:
        if self.target.is_origin():
            return
        
        self.running = True
        
        self.start_pos = self.cur_pos
        
        while not self.drone.connected:
            self.drone.connect()
            self.drone.wait_for_connection(10)
        
        self.drone.start_video()
        
        while self.container is None:
            self.container = av.open(self.drone.get_video_stream())
        
        self.drone.takeoff()

        time.sleep(2)
        
        self.orient_drone()
        
    def shutdown(self) -> None:
        self.running = False
        
        self.drone.backward(0)
        self.drone.land()
        self.drone.quit()
        
        rospy.signal_shutdown()
        
        sys.exit(0)
        
    def check_bounds(self, x_bounds: list, y_bounds: list, z_bounds: list) -> None:
        within_bounds = (x_bounds[0] <= self.x <= x_bounds[1] and  y_bounds[0] <= self.y <= y_bounds[1] and z_bounds[0] <= self.z <= z_bounds[1])
            
        if not within_bounds:
            self.shutdown()
    
    def task_handler(self, pos_arr: list) -> None:
        self.cur_pos = Vector3D.from_arr(pos_arr)
        
        self.check_bounds()
        
        if self.active_task:
            self.active_task()
        
    def plan_path(self, num_pt: int) -> None:
        x_pt = np.linspace(self.start_pos.x, self.target_pos.x, num_pt)
        y_pt = np.linspace(self.start_pos.y, self.target_pos.y, num_pt)
        z_pt = np.linspace(self.start_pos.z, self.target_pos.z, num_pt)
        
        self.waypts = zip(x_pt, y_pt, z_pt)
        self.waypts = [Vector3D.from_arr(arr) for arr in self.waypts]
        
        self.cur_waypt_idx = 0
        
    def reach_waypt(self, waypt_idx, threshold=0.25) -> bool:
        if self.cur_waypt_idx < 0 or waypt_idx >= len(self.waypts):
            return False

        target_waypt = self.waypts[waypt_idx]
        
        diff = self.cur_pos - target_waypt
        
        distance = diff.magnitude()
        
        if distance <= threshold:
            return True
        
        return False
        
    def follow_path(self) -> None:
        if not self.waypts or self.cur_waypt_idx is None:
            raise Exception("Path has not been planned. Please run TelloDrone.plan_path() before doing so.")

        if self.cur_waypt_idx >= len(self.waypts) - 1:
            self.active_task = None
        
        if self.reach_waypt(self.cur_waypt_idx+1):
            self.cur_waypt_idx += 1
            
        target_waypt = self.waypts[self.cur_waypt_idx+1]
        
        self.pid_x.setpoint = target_waypt.x
        self.pid_y.setpoint = target_waypt.y
        self.pid_z.setpoint = target_waypt.z
        
        control_x = self.pid_x(self.cur_pos.x)
        control_y = self.pid_y(self.cur_pos.y)
        control_z = self.pid_z(self.cur_pos.z)
        
        # deal with orientation too btw but thats for next time
        # also fine tune controls for throttle, pitch, yaw
        if control_x < 0:
            self.drone.forward(abs(control_x))
        else:
            self.drone.backward(abs(control_x))
        
        if control_y < 0:
            self.drone.right(abs(control_y))
        else:
            self.drone.left(abs(control_y))
            
        if control_z < 0:
            self.drone.down(abs(control_z))
        else:
            self.drone.up(abs(control_z))
            
        time.sleep(0.5)
    
    def run_objective(self):
        self.startup()
    
        self.plan_path(10)
        
        self.active_task = self.follow_path
        
        if not self.active_task:
            self.shutdown()
