import os
import sys

import time
from datetime import datetime

import logging
import av.container
from log import ColourFormatter

import cv2
import threading

import av

import rospy

from tellopy import Tello

import numpy as np
from vector import Vector3D

from apf import apf

from typing import List, Tuple, Callable

x_bounds = (-0.75, 6.75)
y_bounds = (0.5, 4.5)
z_bounds = (-4.25, -0.75)

logging.basicConfig(level=logging.NOTSET, format='%(asctime)s - %(levelname)s - %(message)s')

class TelloDrone:
    def __init__(self) -> None:
        logging.info("Initialising TelloDrone")
        
        self.cur_pos: Vector3D = Vector3D(0, 0, 0)
        self.target_pos: Vector3D = Vector3D(0, 0, 0)
        
        self.orient_running: bool = False
        
        self.altitude: float = 0
        self.speed: float = 0
        self.battery: float = 0
        self.wifi: float = 0
        self.cam: float = 0
        self.mode: int = 0
        
        self.active_task: Callable = None
        
        self.drone: Tello = Tello()
        self.running: bool = False
        
        self.waypts: List[Vector3D] = []
        self.cur_waypt_idx: int = -1
        
        self.init_time: datetime = datetime.now()
        self.cur_time: datetime = None
        
        log_dir = f"logs/log-{self.init_time.strftime('%d-%m-%Y_%H:%M:%S')}/"
        os.makedirs(log_dir, exist_ok=True)
        
        self.log_info_file: os.PathLike = f"{log_dir}/log-info.log"
        self.log_pos_file: os.PathLike = f"{log_dir}/log-pos.log"
        open(self.log_pos_file, "x")
        
        self.vid_file: os.PathLike = f"vid/vid-{self.init_time.strftime('%d-%m-%Y_%H:%M:%S')}"
        self.video_thread: bool = None
        self.stop_video_thread_event: threading.Event = threading.Event()
        
        self.container: av.container.InputContainer = None
        self.video_writer: cv2.VideoWriter = None
        self.active_vid_task: Callable = None
        
    def start_log(self) -> None:
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
        
    def flight_data_callback(self, event, sender, data) -> None:
        data = str(data).split(" | ")
        data[3] = data[3].replace("  ", " ")
        data = [d.split(": ") for d in data]
        data = {d[0]: d[1] for d in data}
        
        self.altitude = float(data["ALT"])
        self.speed = float(data["SPD"])
        self.battery = float(data["BAT"])
        self.wifi = float(data["WIFI"])
        self.cam = float(data["CAM"])
        self.mode = float(data["MODE"])
        
    def set_target_pos(self, target_pos: Vector3D) -> None:
        logging.info(f"Setting target position: {target_pos}")
        self.target_pos = target_pos
        self.waypts = [None, self.target_pos]
        self.cur_waypt_idx = 0
        
    def orient_drone(self, threshold=0.25) -> None:
        logging.info("Orienting the drone to face negative-x")
        x_diff = 0
        y_diff = 1
        
        while y_diff > threshold:
            start_pos = self.cur_pos
            self.drone.forward(30)
            time.sleep(3)
            end_pos = self.cur_pos
            
            pos_diff = end_pos - start_pos
            x_diff = pos_diff.x
            y_diff = pos_diff.y
            
            logging.debug(f"X-Diff : {x_diff}")
            logging.debug(f"Y-Diff : {y_diff}")
            
            turn_val = 30
            
            if x_diff > 0:
                turn_val += 15
            elif y_diff <= threshold:
                return
            
            if y_diff > 0:
                self.drone.clockwise(turn_val)
            else:
                self.drone.counter_clockwise(turn_val)
                
            time.sleep(0.5)
            self.drone.clockwise(0)

    def startup(self) -> None:
        self.start_log()
        
        if self.target_pos.is_origin():
            logging.warning("Target position is the origin. Aborting startup.")
            self.shutdown(error=True, reason="Target position not set")

        logging.info("Starting up the TelloDrone")

        logging.info("Attempting to connect to the drone")
        self.drone.connect()
        self.drone.wait_for_connection(10)
        
        if not self.drone.connected:
            self.shutdown(error=True, reason="Drone not connected")
        
        self.drone.subscribe(self.drone.EVENT_FLIGHT_DATA, self.flight_data_callback)
        
        logging.info("Drone connected successfully")
        
        self.drone.start_video()
        
        while self.container is None:
            logging.info("Opening video stream from the drone")
            self.container = av.open(self.drone.get_video_stream())
            

        logging.info("Taking off")
        self.drone.takeoff()
        time.sleep(2)
        
        self.start_pos = self.cur_pos
        self.waypts[0] = self.start_pos
        
        self.start_video_thread()
        
        self.running = True

    def shutdown(self, error=False, reason=None) -> None:
        if error:
            logging.error(reason)
        logging.info("Shutting down all processes")
        
        self.drone.backward(0)
        
        while self.altitude > 0:
            self.drone.land()
            time.sleep(1)
        
        self.stop_video_thread()

        logging.info("Waiting for tasks to finish...")

        logging.info("Shutting down drone and ROS node")
        
        self.drone.quit()
        rospy.signal_shutdown("Failed" if error else "Objective Completed")
        
        if error:
            logging.critical(reason)
        else:
            logging.info(reason if reason else "Objective Completed")
        
        sys.exit(0)

    def check_bounds(self, x_bounds: tuple, y_bounds: tuple, z_bounds: tuple) -> None:
        inst_pos = self.cur_pos
        x_within_bounds = (x_bounds[0] <= inst_pos.x and inst_pos.x <= x_bounds[1])
        y_within_bounds = (y_bounds[0] <= inst_pos.y and inst_pos.y <= y_bounds[1])
        z_within_bounds = (z_bounds[0] <= inst_pos.z and inst_pos.z <= z_bounds[1])

        if not x_within_bounds:
            logging.warning("Drone position-x is out of bounds.")
            self.shutdown(error=True, reason=f"Drone position-x is out of bounds : {self.cur_pos}")
        if not y_within_bounds:
            logging.warning("Drone position-y is out of bounds.")
            self.shutdown(error=True, reason=f"Drone position-y is out of bounds : {self.cur_pos}")
        if not z_within_bounds:
            logging.warning("Drone position-z is out of bounds.")
            self.shutdown(error=True, reason=f"Drone position-z is out of bounds : {self.cur_pos}")
            
    def process_frame(self, frame: av.VideoFrame):
        img = frame.to_ndarray(format='bgr24')                    
        self.video_writer.write(img)
        
        if self.active_vid_task:
            self.active_vid_task(frame)
    
    def process_video(self) -> None:
        logging.info("Processing video frames in thread.")
        
        while not self.stop_video_thread_event.is_set():
            try:
                for frame in self.container.decode(video=0):
                    self.process_frame(frame)
                    
                    if self.stop_video_thread_event.is_set():
                        break
                    
            except Exception as e:
                logging.error(f"Error in video processing: {e}")
                break

        logging.info("Exiting video processing thread.")

    def start_video_thread(self) -> None:
        if self.container is None:
            logging.error("Video stream not initialized. Cannot start video thread.")
            return

        if self.video_thread and self.video_thread.is_alive():
            logging.warning("Video thread is already running.")
            return

        if not self.video_writer:
            fourcc = cv2.VideoWriter.fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(f"{self.vid_file}.avi", fourcc, 24, (960, 720))

        self.stop_video_thread_event.clear()
        self.video_thread = threading.Thread(target=self.process_video)
        self.video_thread.start()
        logging.info("Video processing thread started.")

    def stop_video_thread(self) -> None:
        if not self.video_thread or not self.video_thread.is_alive():
            logging.warning("No video thread to stop.")
            return

        self.stop_video_thread_event.set()
        self.video_thread.join()
        self.video_writer.release()
        logging.info("Video processing thread stopped, and video file saved.")

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

    def get_dist_waypt(self, waypt_idx) -> float:
        if self.cur_waypt_idx < 0 or waypt_idx >= len(self.waypts):
            logging.warning("Waypoint index out of range")
            return False

        target_waypt = self.waypts[waypt_idx]
        diff = self.cur_pos - target_waypt
        distance = diff.magnitude()
        
        return distance

    def follow_path(self) -> None:
        if not self.waypts:
            logging.error("Path not planned. Call TelloDrone.set_target_pos() and/or TelloDrone.plan_path() first.")

        if self.cur_waypt_idx >= len(self.waypts) - 1:
            self.active_task = None
        
        dist_waypt = self.get_dist_waypt(self.cur_waypt_idx + 1)
        if dist_waypt <= 0.5:
            self.cur_waypt_idx += 1
            logging.info("Drone has reached waypoint")
            self.active_task = None

        target_waypt = self.waypts[self.cur_waypt_idx + 1]
        
        attract_coeff = 60
        if dist_waypt < 2.0:
            attract_coeff = attract_coeff // 3 * dist_waypt
        
        control_x, control_y, control_z = apf(self.cur_pos, self.target_pos, attract_coeff)

        # Logging control signals
        logging.debug(f"Attraction Coefficient : {attract_coeff}")
        logging.debug(f"Control signals: X={control_x}, Y={control_y}, Z={control_z}")

        # assuming facing towards negative-x
        if control_x < 0:
            self.drone.forward(abs(control_x))
        else:
            self.drone.backward(abs(control_x))

        if control_y > 0:
            self.drone.right(abs(control_y))
        else:
            self.drone.left(abs(control_y))

        if control_z < 0:
            self.drone.down(abs(control_z))
        else:
            self.drone.up(abs(control_z))

        time.sleep(0.3)
        logging.info("Following path")
        logging.debug(f"Current position : {self.cur_pos}")
        logging.debug(f"Target position : {target_waypt}")

    def run_objective(self) -> None:
        logging.info("Running objective")
        self.startup()
        
        # self.plan_path(3)
        
        self.active_task = self.follow_path
        
        if self.running and self.active_task is None:
            self.shutdown()