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

import json

import rospy

from tellopy import Tello

import numpy as np
from vector import Vector3D

from apf import apf
# from test_apf import apf

from PIL import Image

from typing import List, Callable

x_bounds = (-0.75, 6.85)
y_bounds = (0, 4.5)
z_bounds = (-4.25, 0.0)

logging.basicConfig(level=logging.NOTSET, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Loading Model")
from depth_model import estimate_depth

class TelloDrone:
    def __init__(self) -> None:
        logging.info("Initialising TelloDrone")
        
        self.takeoff_pos: Vector3D = Vector3D(0, 0, 0)
        self.start_pos: Vector3D = Vector3D(0, 0, 0)
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
        
        self.init_time: datetime = datetime.now()
        self.cur_time: datetime = None
        
        self.run_name = self.init_time.strftime('%d-%m-%Y_%H:%M:%S')
        
        self.log_dir = f"logs/log-{self.run_name}/"
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.log_info_file: os.PathLike = f"{self.log_dir}/log-info.log"
        self.log_pos_file: os.PathLike = f"{self.log_dir}/log-pos.log"
        open(self.log_pos_file, "x")
        
        self.vid_file: os.PathLike = f"vid/vid-{self.run_name}"
        self.video_thread: threading.Thread = None
        self.stop_video_thread_event: threading.Event = threading.Event()
        
        self.container: av.container.InputContainer = None
        self.video_writer: cv2.VideoWriter = None
        self.active_vid_task: Callable = None
        
        self.frame_idx = -1
        
        os.makedirs(f"img/original/{self.init_time}", exist_ok=True)
        os.makedirs(f"img/depth/{self.init_time}", exist_ok=True)
        
        self.obstacles = []
        
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
        
        if self.battery <= 10:
            logging.critical(f"Drone battery very low at {self.battery}%")
            self.shutdown(error=True, reason=f"Insufficient battery at {self.battery}%")
        elif self.battery <= 20:
            logging.warning(f"Drone battery low at {self.battery}%")
        
    def set_target_pos(self, target_pos: Vector3D) -> None:
        logging.info(f"Setting target position: {target_pos}")
        self.target_pos = target_pos
        
    def set_obstacles(self, obstacles: list) -> None:
        self.obstacles.extend(obstacles)
        
    def save_log_config(self) -> None:
        log_config_dir = f"{self.log_dir}/log-config.json"
        
        config = {
            "takeoff_pos": self.takeoff_pos.to_arr(),
            "start_pos": self.start_pos.to_arr(),
            "end_pos": self.cur_pos.to_arr(),
            "target_pos": self.target_pos.to_arr(),
            "obstacles": [(obp.to_arr(), obr) for obp, obr in self.obstacles]
        }
        
        with open(log_config_dir, "w+") as f:
            f.write(json.dumps(config, indent=4))

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
            
        self.takeoff_pos = self.cur_pos

        logging.info("Taking off")
        self.drone.takeoff()
        
        self.start_video_thread()
        
        time.sleep(2)
        self.start_pos = self.cur_pos
        
        self.running = True

    def shutdown(self, error=False, reason=None) -> None:
        if error:
            logging.error(reason)
            
        logging.info("Shutting down all processes")
        
        self.stop_video_thread()
        
        if not os.path.exists(f"{self.log_dir}/log-config.json"):
            self.save_log_config()
        
        self.drone.backward(0)
        
        self.drone.land()
        time.sleep(1)

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
            
    def run_depth_model(self, frame_img: np.ndarray):
        if self.frame_idx % 20 == 0:
            logging.info("Video frame captured")
            logging.info(f"Estimating depth of frame {self.frame_idx}")
            depth_image = estimate_depth(frame_img)
            
            orig_output_path = os.path.join(f"img/original/{self.init_time}", f"frame-{self.frame_idx}.png")
            depth_output_path = os.path.join(f"img/depth/{self.init_time}", f"frame-{self.frame_idx}.png")
            
            Image.fromarray(frame_img).save(orig_output_path)
            Image.fromarray(depth_image).save(depth_output_path)
    
    def process_frame(self, frame: av.VideoFrame):
        img = frame.to_ndarray(format='bgr24')                    
        self.video_writer.write(img)
        
        if self.frame_idx >= 100 and self.active_vid_task:
            self.active_vid_task(img)
    
    def process_video(self) -> None:
        logging.info("Processing video frames in thread.")
        
        while not self.stop_video_thread_event.is_set():
            try:
                for frame in self.container.decode(video=0):
                    self.frame_idx += 1
                    
                    if self.stop_video_thread_event.is_set():
                        break
                    
                    self.process_frame(frame)
                    
            except Exception as e:
                logging.error(f"Error in video processing: {e}")

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


    def follow_path(self) -> None:
        if not self.target_pos:
            logging.error("Path not planned. Call TelloDrone.plan_path() first.")
        
        self.active_vid_task = self.run_depth_model
        local_delta = (self.cur_pos - self.target_pos).magnitude()
        
        if local_delta <= 0.4:
            logging.info("Drone has reached target")
            self.active_vid_task = None
            self.active_task = None
        
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

        # Logging control signals
        logging.debug(f"Resultant Force : {total_force}")
        logging.debug(f"Attractive Force : {attract_force}")
        logging.debug(f"Repulsive Force : {repel_force}")
        logging.debug(f"Control signals: X={velocity_x}, Y={velocity_y}, Z={velocity_z}")

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
        logging.debug(f"Current position : {self.cur_pos}")
        logging.debug(f"Target position : {self.target_pos}")

    def run_objective(self) -> None:
        logging.info("Running objective")
        self.startup()
        
        self.active_task = self.follow_path
        
        if self.running and self.active_task is None:
            self.shutdown()