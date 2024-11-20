import os
import sys

import time
from datetime import datetime

import logging
from log import ColourFormatter

import cv2
import threading

import av.container
import rospy

from tellopy import Tello

import numpy as np
from vector import Vector3D

import av

from simple_pid import PID

x_bounds = (0.3, 6.25)
y_bounds = (0.5, 4.5)
z_bounds = (-4.25, -1.25)

logging.basicConfig(level=logging.NOTSET, format='%(asctime)s - %(levelname)s - %(message)s')

class TelloDrone:
    def __init__(self) -> None:
        logging.info("Initialising TelloDrone")
        
        self.cur_pos = Vector3D(0, 0, 0)
        self.target_pos = Vector3D(0, 0, 0)
        self.orientation = None
        
        self.altitude = 0
        self.speed = 0
        self.battery = 0
        self.wifi = 0
        self.cam = 0
        self.mode = 0
        
        self.active_task = lambda: None
        
        self.drone = Tello()
        self.running = False
        
        self.waypts = []
        self.cur_waypt_idx = -1
        
        self.container = None
        self.cur_frame = None
        
        self.pid_x = PID(Kp=1.0, Ki=0.1, Kd=0.05, output_limits=(-10, 10))
        self.pid_y = PID(Kp=1.0, Ki=0.1, Kd=0.05, output_limits=(-10, 10))
        self.pid_z = PID(Kp=1.0, Ki=0.1, Kd=0.05, output_limits=(-10, 10))
        
        self.init_time = datetime.now()
        self.cur_time = None
        
        log_dir = f"logs/log-{self.init_time.strftime('%d-%m-%Y_%H:%M:%S')}/"
        os.makedirs(log_dir, exist_ok=True)
        
        self.log_info_file = f"{log_dir}/log-info.log"
        self.log_pos_file = f"{log_dir}/log-pos.log"
        
        open(self.log_pos_file, "x")
        
    def start_log(self):
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.NOTSET)
        stream_handler.setFormatter(ColourFormatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        file_handler = logging.FileHandler(self.log_info_file)
        file_handler.setLevel(logging.NOTSET)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        logger = logging.getLogger()
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)
        logger.setLevel(logging.NOTSET)
        
    def flight_data_callback(self, event, sender, data):
        data = str(data).split(" | ")
        data[3] = data[3].replace("  ", " ")
        data = [d.split(": ") for d in data]
        data = {d[0]: d[1] for d in data}
        
        self.altitude = data["ALT"]
        self.speed = data["SPD"]
        self.battery = data["BAT"]
        self.wifi = data["WIFI"]
        self.cam = data["CAM"]
        self.mode = data["MODE"]
        
    def set_target_pos(self, target_pos: Vector3D) -> None:
        logging.info(f"Setting target position: {target_pos}")
        self.target_pos = target_pos

    def orient_drone(self) -> None:
        # do this some other time
        logging.info("Orienting the drone")
        f_pos = self.cur_pos
        self.drone.forward(30)
        time.sleep(3)
        s_pos = self.cur_pos

        x_diff = f_pos.x - s_pos.x
        y_diff = f_pos.y - s_pos.y

        self.orientation = np.arctan2(y_diff, x_diff)
        self.orientation = np.degrees(self.orientation)

        if self.orientation < -180:
            self.orientation += 360
        elif self.orientation > 180:
            self.orientation -= 360

        logging.info(f"Drone orientation set to: {self.orientation} degrees")

    def startup(self) -> None:
        self.start_log()
        
        if self.target_pos.is_origin():
            logging.warning("Target position is the origin. Aborting startup.")
            self.shutdown(error=True, reason="Target position not set")

        logging.info("Starting up the TelloDrone")
        self.running = True
        self.start_pos = self.cur_pos

        logging.info("Attempting to connect to the drone")
        self.drone.connect()
        self.drone.wait_for_connection(10)
        
        self.drone.subscribe(self.drone.EVENT_FLIGHT_DATA, self.flight_data_callback)
        
        time.sleep(2)

        # logging.info("Drone connected successfully")
        # self.drone.start_video()

        # while self.container is None:
        #     logging.info("Opening video stream from the drone")
        #     self.container = av.open(self.drone.get_video_stream())

        logging.info("Taking off")
        self.drone.takeoff()
        time.sleep(2)

    def shutdown(self, error=False, reason=None) -> None:
        logging.info("Shutting down all processes")
        
        while True:
            self.running = False
            
            logging.info("Tello Drone shutdown")
            self.drone.backward(0)
            self.drone.land()
            self.drone.quit()

            logging.info("ROS node shutdown")
            rospy.signal_shutdown("Failed")
            if error:
                logging.critical(reason)
            else:
                logging.info(reason if reason else "Objective Completed")
            sys.exit(0)

    def check_bounds(self, x_bounds: tuple, y_bounds: tuple, z_bounds: tuple) -> None:
        # logging.debug(f"Checking bounds: x={x_bounds}, y={y_bounds}, z={z_bounds}")
        within_bounds = (x_bounds[0] <= self.cur_pos.x and self.cur_pos.x <= x_bounds[1] and y_bounds[0] <= self.cur_pos.y and self.cur_pos.y <= y_bounds[1] and z_bounds[0] <= self.cur_pos.z and self.cur_pos.z <= z_bounds[1])

        if not within_bounds:
            logging.warning("Drone is out of bounds.")
            self.shutdown(error=True, reason=f"Drone out of bounds : {self.cur_pos}")

    # def display_video(self) -> None:
    #     pass
    
    # def process_video(self) -> None:
    #     logging.info("Processing video frames")
    #     for frame in self.container.decode(video=0):
    #         img = frame.to_ndarray(format='bgr24')
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #         self.cur_frame = np.asarray(img[:, :])

    # def video_handler(self) -> None:
    #     if not self.running or self.container is None:
    #         logging.error("TelloDrone has not started running. Call TelloDrone.startup() first.")
    #         raise Exception("TelloDrone has not started running. Please run TelloDrone.startup() to continue.")

    def task_handler(self, pos_arr: list) -> None:
        self.cur_pos = Vector3D.from_arr(pos_arr)
        self.cur_time = datetime.now()
        
        delta_time = self.cur_time - self.init_time
        
        with open(self.log_pos_file, "a") as f:
            f.write(f"{self.cur_time} {delta_time.total_seconds()} {self.cur_pos.x} {self.cur_pos.y} {self.cur_pos.z}" + "\n")
        
        self.check_bounds(x_bounds, y_bounds, z_bounds)
        
        if self.active_task is not None:
            self.active_task()
        elif self.running:
            self.shutdown()

    def plan_path(self, num_pt: int) -> None:
        logging.info(f"Planning path with {num_pt} waypoints")
        x_pt = np.linspace(self.start_pos.x, self.target_pos.x, num_pt)
        y_pt = np.linspace(self.start_pos.y, self.target_pos.y, num_pt)
        z_pt = np.linspace(self.start_pos.z, self.target_pos.z, num_pt)
        
        self.waypts = zip(x_pt, y_pt, z_pt)
        self.waypts = [Vector3D.from_arr(arr) for arr in self.waypts]
        self.cur_waypt_idx = 0
        logging.info("Path planning complete")
        
        for idx, waypt in enumerate(self.waypts):
            logging.debug(f"Waypoint {idx} : {waypt}")

    def reach_waypt(self, waypt_idx, threshold=1) -> bool:
        if self.cur_waypt_idx < 0 or waypt_idx >= len(self.waypts):
            logging.warning("Waypoint index out of range")
            return False

        target_waypt = self.waypts[waypt_idx]
        diff = self.cur_pos - target_waypt
        distance = diff.magnitude()

        if distance <= threshold:
            logging.info(f"Reached waypoint {waypt_idx}")
            return True

        return False

    def follow_path(self) -> None:
        if not self.waypts or self.cur_waypt_idx is None:
            logging.error("Path not planned. Call TelloDrone.plan_path() first.")
            # raise Exception("Path has not been planned. Please run TelloDrone.plan_path() before doing so.")

        if self.cur_waypt_idx >= len(self.waypts) - 1:
            self.active_task = None

        if self.reach_waypt(self.cur_waypt_idx + 1):
            self.cur_waypt_idx += 1

        target_waypt = self.waypts[self.cur_waypt_idx + 1]

        self.pid_x.setpoint = target_waypt.x
        self.pid_y.setpoint = target_waypt.y
        self.pid_z.setpoint = target_waypt.z

        control_x = self.pid_x(self.cur_pos.x)
        control_y = self.pid_y(self.cur_pos.y)
        control_z = self.pid_z(self.cur_pos.z)

        # Logging control signals
        logging.info(f"Control signals: X={control_x}, Y={control_y}, Z={control_z}")
        
        magnitude = 4

        # assuming facing towards negative-x
        if control_x < 0:
            self.drone.forward(abs(control_x * magnitude))
        else:
            self.drone.backward(abs(control_x * magnitude))

        if control_y > 0:
            self.drone.right(abs(control_y * magnitude / 2))
        else:
            self.drone.left(abs(control_y * magnitude / 2))

        if control_z < 0:
            self.drone.down(abs(control_z * magnitude / 2))
        else:
            self.drone.up(abs(control_z * magnitude / 2))

        time.sleep(1)
        logging.info("Following path")
        logging.debug(f"Current position : {self.cur_pos}")
        logging.debug(f"Target position : {target_waypt}")

    def run_objective(self):
        logging.info("Running objective")
        self.startup()
        self.set_target_pos(self.cur_pos - Vector3D(1, 0, 0))
        
        self.plan_path(10)
        
        self.active_task = self.follow_path