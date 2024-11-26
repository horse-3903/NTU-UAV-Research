import os
import math
import cv2

import threading

import pygame

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tellodrone.core import TelloDrone

if TYPE_CHECKING:
    from tellodrone.core import TelloDrone

def setup_display(self: "TelloDrone"):
    pygame.init()

    # Set up the display window
    self.screen = pygame.display.set_mode((960, 720))
    pygame.display.set_caption("Tello Drone Video Feed")

    # Load the camera icon image
    camera_icon = pygame.image.load("camera-icon.png")
    icon_size = (30, 30)
    camera_icon = pygame.transform.scale(camera_icon, icon_size)

    # Button properties
    button_radius = 30
    button_center = (button_radius + 10, button_radius + 10)
    button_color = (50, 150, 250)
    button_hover_color = (80, 180, 255)

    self.clock = pygame.time.Clock()
    self.display_running = True

    os.makedirs(f"calibrate/img/{self.init_time}", exist_ok=True)

    def draw_circle_button(surface, center, radius, color, hover=False):        
        button_color_to_use = button_hover_color if hover else color
        pygame.draw.circle(surface, button_color_to_use, center, radius)
        
        icon_pos = (center[0] - icon_size[0] // 2, center[1] - icon_size[1] // 2)
        surface.blit(camera_icon, icon_pos)

    def display_loop():
        self.logger.info("Pygame display loop started.")

        while self.display_running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.display_running = False
                    break
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = event.pos
                    distance = math.sqrt((mouse_pos[0] - button_center[0])**2 + (mouse_pos[1] - button_center[1])**2)
                    if distance <= button_radius:
                        self.process_image()

            if self.cur_frame is not None:
                frame_surface = pygame.surfarray.make_surface(cv2.cvtColor(self.cur_frame, cv2.COLOR_BGR2RGB))
                self.screen.blit(pygame.transform.rotate(frame_surface, -90), (0, 0))

            mouse_pos = pygame.mouse.get_pos()
            is_hovering = math.sqrt((mouse_pos[0] - button_center[0])**2 + (mouse_pos[1] - button_center[1])**2) <= button_radius
            draw_circle_button(self.screen, button_center, button_radius, button_color, hover=is_hovering)

            pygame.display.flip()
            self.clock.tick(30)

        pygame.quit()
        self.logger.info("Pygame display loop exited.")

    self.display_thread = threading.Thread(target=display_loop)
    self.display_thread.start()


def save_image(self: "TelloDrone", dir: os.PathLike = "img/manual"):
    if self.cur_frame is not None:
        img_path = f"{dir}/{self.init_time}/frame-{self.frame_idx}.jpg"
        cv2.imwrite(img_path, self.cur_frame)
        self.logger.info(f"Image saved: {img_path}")
    else:
        self.logger.warning("No frame available to save.")


def process_image(self: "TelloDrone"):
    if self.cur_frame is not None:
        self.logger.info(f"Processing Image")
        
        self.save_image()
        
        if self.active_img_task and (self.active_img_task_thread is None or not self.active_img_task_thread.is_alive()):
            def task_wrapper():
                try:
                    self.active_img_task()
                except Exception as e:
                    self.logger.error(f"Error in active video task: {e}")
                finally:
                    self.active_vid_task_thread = None

            self.active_img_task_thread = threading.Thread(target=task_wrapper)
            self.active_img_task_thread.start()
    else:
        self.logger.warning("No frame available process.")
        

def process_frame(self: "TelloDrone"):
    self.video_writer.write(self.cur_frame)

    if self.frame_idx < 100:
        return
    
    if self.active_vid_task and (self.active_vid_task_thread is None or not self.active_vid_task_thread.is_alive()):
        def task_wrapper():
            try:
                self.active_vid_task()
            except Exception as e:
                self.logger.error(f"Error in active video task: {e}")
            finally:
                self.active_vid_task_thread = None

        self.active_vid_task_thread = threading.Thread(target=task_wrapper)
        self.active_vid_task_thread.start()


def process_video(self: "TelloDrone") -> None:
    self.logger.info("Processing video frames in thread.")
    
    while not self.stop_video_thread_event.is_set():
        try:
            for frame in self.container.decode(video=0):
                self.frame_idx += 1
                
                if self.stop_video_thread_event.is_set():
                    break
                
                self.cur_frame = frame.to_ndarray(format="bgr24")
                self.process_frame()
                
        except Exception as e:
            self.logger.error(f"Error in video processing: {e}")

    self.logger.info("Exiting video processing thread.")
    

def start_video_thread(self: "TelloDrone") -> None:
    if self.container is None:        
        self.logger.error("Video stream not initialized. Cannot start video thread.")

    if self.video_thread and self.video_thread.is_alive():
        self.logger.warning("Video thread is already running.")
        return

    fourcc = cv2.VideoWriter.fourcc(*'XVID')
    self.video_writer = cv2.VideoWriter(f"{self.vid_file}.avi", fourcc, 24, (960, 720))

    self.stop_video_thread_event.clear()
    self.video_thread = threading.Thread(target=self.process_video)
    self.video_thread.start()
    self.logger.info("Video processing thread started.")


def stop_video_thread(self: "TelloDrone") -> None:
    if not self.video_thread or not self.video_thread.is_alive():
        self.logger.warning("No video thread to stop.")
        return
    
    self.stop_video_thread_event.set()
    
    self.video_writer.release()
    
    # Stop active tasks (if any)
    if self.active_vid_task_thread and self.active_vid_task_thread.is_alive():
        self.logger.info("Waiting for active video task thread to finish...")
        self.active_vid_task_thread.join()
    
    # Ensure that the video thread is stopped properly
    if self.video_thread and self.video_thread.is_alive():
        self.logger.info("Waiting for video thread to finish...")
        self.video_thread.join()

    # Ensure display thread is also stopped
    if self.display_thread and self.display_thread.is_alive():
        self.logger.info("Waiting for display thread to finish...")
        self.display_thread.join()
    
    self.logger.info("Video processing thread stopped, and video file saved.")