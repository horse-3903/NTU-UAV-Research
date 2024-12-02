import os
import sys

import av.container
import rospy

import time
from datetime import datetime

from tellopy import Tello

from vector import Vector3D

from functools import partial

import av
import pygame
import logging
from numpy import ndarray
from typing import NoReturn, List, Tuple, Callable
from threading import Thread, Event
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
        self.x_bounds = (-0.50, 7.00)
        self.y_bounds = (-0.50, 4.50)
        self.z_bounds = (-3.75, -0.50)
        self.obstacles: List[Tuple[Vector3D, float]] = []
        
        # task
        self.active_task : Callable = None

        # run name
        self.run_name = self.init_time.strftime('%Y-%m-%d_%H:%M:%S')
        
        # flight information
        self.altitude = 0
        self.speed = 0
        self.battery = 0
        self.wifi = 0
        self.cam = 0
        self.mode = 0
        
        # video 
        self.vid_file = f"vid/raw/vid-{self.run_name}"
        self.video_thread = Thread()
        self.active_vid_task_thread = Thread()
        self.active_img_task_thread = Thread()
        self.stop_video_thread_event = Event()
        
        self.container: av.container.InputContainer = None
        self.video_writer = VideoWriter()
        
        self.active_vid_task: Callable = None
        self.active_img_task: Callable = None
        
        self.cur_frame_idx = -1
        self.cur_frame : ndarray = None
        
        self.display_running = False
        self.screen: pygame.Surface = None
        self.clock = pygame.time.Clock()
        self.display_thread = Thread()
        
        os.makedirs(f"img/original/{self.init_time}", exist_ok=True)
        os.makedirs(f"img/depth/{self.init_time}", exist_ok=True)
        os.makedirs(f"img/annotated/{self.init_time}", exist_ok=True)
        os.makedirs(f"img/manual/{self.init_time}", exist_ok=True)
        
        # logging
        self.log_dir = f"logs/log-{self.run_name}/"
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.log_info_file = f"{self.log_dir}/log-info.log"
        self.log_pos_file = f"{self.log_dir}/log-pos.log"
        self.log_config_file = f"{self.log_dir}/log-config.json"
        
        self.logger = logging.getLogger()
        
        # depth model
        self.model_name = "model/zoedepth-nyu-kitti"
        self.image_processor: ZoeDepthImageProcessor = None
        self.depth_model: ZoeDepthForDepthEstimation = None
        self.depth_model_run = False

        
    # importing functions
    from tellodrone.log import setup_logging, save_log_config
    from tellodrone.flight_control import flight_data_callback, check_bounds
    from tellodrone.video import setup_display, process_image, process_frame, process_video, start_video_thread, stop_video_thread, save_calibrate_image
    from tellodrone.task import task_handler
    from tellodrone.follow_path import set_target_pos, add_obstacle, follow_path
    from tellodrone.depth_model import load_depth_model, run_depth_model, estimate_depth

    
    def startup_video(self) -> None:
        self.logger.info("Attempting to connect to the drone")
        self.drone.connect()
        self.drone.wait_for_connection(10.0)
        
        self.logger.info("Loading Depth Model")
        self.load_depth_model()
        
        self.logger.info("Starting Drone Video")
        self.drone.start_video()
        
        while self.container is None:
            self.logger.info("Opening video stream from the drone")
            self.drone.start_video()
            self.container = av.open(self.drone.get_video_stream())
            
        self.start_video_thread()
        self.setup_display()


    def startup(self, display: bool) -> None:            
        self.logger.info("Loading Depth Model")
        self.load_depth_model()

        self.logger.info("Starting up the TelloDrone")
        
        self.logger.info("Attempting to connect to the drone")
        self.drone.connect()
        self.drone.wait_for_connection(10.0)
        
        if not self.drone.connected:
            self.shutdown(error=True, reason="Drone not connected")
        
        self.drone.subscribe(self.drone.EVENT_FLIGHT_DATA, self.flight_data_callback)
        
        self.logger.info("Drone connected successfully")
        
        self.logger.info("Starting Drone Video")
        self.drone.start_video()

        time.sleep(1)
        
        while self.container is None:
            self.logger.info("Opening video stream from the drone")
            self.container = av.open(self.drone.get_video_stream())
            
        self.start_video_thread()
        
        if display:
            self.setup_display()
        
        self.takeoff_pos = self.cur_pos

        self.logger.info("Taking off")
        self.drone.takeoff()
        
        time.sleep(2)
        self.start_pos = self.cur_pos
        
        self.running = True


    def shutdown(self, error=False, reason=None) -> NoReturn:
        if error:
            self.logger.error(reason)
        
        self.logger.info(f"Shutting down all processes : {reason}")
        
        self.save_log_config()
        
        self.logger.info("Landing Drone")
        self.drone.backward(0)
        
        self.drone.land()
        
        self.logger.info("Waiting for tasks to finish...")
        self.logger.info("Shutting down drone and ROS node")
        
        self.display_running = False
        self.stop_video_thread()
        
        pygame.quit()
        self.drone.quit()
        rospy.signal_shutdown("Failed" if error else "Objective Completed")
        
        if error:
            self.logger.critical(reason)
        else:
            self.logger.info(reason if reason else "Objective Completed")
        
        sys.exit(0)
        
        
    def run_calibration(self) -> None:
        self.setup_logging()
        self.logger.info("Running objective")
        
        self.active_img_task = partial(self.save_calibrate_image, manual=True)
        
        self.startup_video()
    
        
    def run_objective(self, display: bool = False) -> None:
        self.setup_logging()
        self.logger.info("Running objective")
        
        self.active_task = self.follow_path
        # self.active_task = partial(time.sleep, 1)
        
        self.startup(display=display)
        # self.startup_video()