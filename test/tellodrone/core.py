import os
import sys

import rospy

import time
from datetime import datetime

from tellopy import Tello

from tellodrone.vector import Vector3D

from typing import NoReturn
from logging import Logger
from threading import Thread, Event
from av.container import InputContainer
from cv2 import VideoWriter
from transformers import ZoeDepthForDepthEstimation, ZoeDepthImageProcessor

class TelloDrone:
    def __init__(self) -> None:
        # position
        self.takeoff_pos = Vector3D(0, 0, 0)
        self.start_pos = Vector3D(0, 0, 0)
        self.cur_pos = Vector3D(0, 0, 0)
        self.target_pos = Vector3D(0, 0, 0)

        # time
        self.init_time = datetime.now()
        self.cur_time = datetime.now()
        
        # drone class
        self.drone = Tello()
        self.running = False
        
        # bounds
        self.x_bounds = (-0.75, 6.85)
        self.y_bounds = (0, 4.5)
        self.z_bounds = (-4.25, 0.0)
        
        # task
        self.active_task = lambda: None

        # run name
        self.run_name = self.init_time.strftime('%d-%m-%Y_%H:%M:%S')
        
        # flight information
        self.altitude = 0.0
        self.speed = 0.0
        self.battery = 0.0
        self.wifi = 0.0
        self.cam = 0.0
        self.mode = 0.0
        
        # depth model
        self.image_processor = ZoeDepthImageProcessor()
        self.depth_model = ZoeDepthForDepthEstimation()
        self.load_depth_model()
        
        # video 
        self.vid_file: os.PathLike = f"vid/vid-{self.run_name}"
        self.video_thread = Thread()
        self.stop_video_thread_event = Event()
        
        self.container = InputContainer()
        self.video_writer = VideoWriter()
        self.active_vid_task = lambda: None
        
        self.frame_idx = -1
        
        os.makedirs(f"img/original/{self.init_time}", exist_ok=True)
        os.makedirs(f"img/depth/{self.init_time}", exist_ok=True)
        
        # logging
        self.log_dir = f"logs/log-{self.run_name}/"
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.log_info_file = f"{self.log_dir}/log-info.log"
        self.log_pos_file = f"{self.log_dir}/log-pos.log"
        self.log_config_file = f"{self.log_dir}/log-config.json"
        
        self.logger = Logger(None)
        
        self.setup_logging()
        
    # importing functions
    from log import setup_logging, save_log_config
    from flight_control import flight_data_callback, check_bounds
    from video import process_frame, process_video, start_video_thread, stop_video_thread
    from task import task_handler, run_objective
    from follow_path import set_target_pos
    from depth_model import load_depth_model, run_depth_model, estimate_depth

    def startup(self) -> None:        
        if self.target_pos.is_origin():
            self.logger.warning("Target position is the origin. Aborting startup.")
            self.shutdown(error=True, reason="Target position not set")

        self.logger.info("Starting up the TelloDrone")

        self.logger.info("Attempting to connect to the drone")
        self.drone.connect()
        self.drone.wait_for_connection(10)
        
        if not self.drone.connected:
            self.shutdown(error=True, reason="Drone not connected")
        
        self.drone.subscribe(self.drone.EVENT_FLIGHT_DATA, self.flight_data_callback)
        
        self.logger.info("Drone connected successfully")
        
        self.drone.start_video()
            
        self.takeoff_pos = self.cur_pos

        self.logger.info("Taking off")
        self.drone.takeoff()
        
        self.start_video_thread()
        
        time.sleep(2)
        self.start_pos = self.cur_pos
        
        self.running = True

    def shutdown(self, error=False, reason=None) -> NoReturn:
        if error:
            self.logger.error(reason)
            
        self.logger.info("Shutting down all processes")
        
        self.stop_video_thread()
        
        self.save_log_config()
        
        self.drone.backward(0)
        
        self.drone.land()
        time.sleep(1)

        self.logger.info("Waiting for tasks to finish...")
        self.logger.info("Shutting down drone and ROS node")
        
        self.drone.quit()
        rospy.signal_shutdown("Failed" if error else "Objective Completed")
        
        if error:
            self.logger.critical(reason)
        else:
            self.logger.info(reason if reason else "Objective Completed")
        
        sys.exit(0)