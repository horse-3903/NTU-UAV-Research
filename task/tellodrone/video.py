import av
import cv2

import threading

import pygame

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tellodrone.core import TelloDrone
    
import os
import pygame
import cv2
import threading
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tellodrone.core import TelloDrone


def setup_display(self: "TelloDrone"):
    pygame.init()

    # Set up the display window
    self.screen = pygame.display.set_mode((960, 720))  # Resolution of the Tello drone feed
    pygame.display.set_caption("Tello Drone Video Feed")

    # Fonts and colors for the button
    font = pygame.font.Font(None, 36)
    button_color = (50, 150, 50)
    button_hover_color = (80, 200, 80)
    text_color = (255, 255, 255)
    button_rect = pygame.Rect(10, 10, 250, 50)  # Position and size of the button

    self.clock = pygame.time.Clock()
    self.display_running = True

    # Create calibration image directory based on initialization time
    self.init_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(f"calibrate/img/{self.init_time}", exist_ok=True)

    def display_loop():
        self.logger.info("Pygame display loop started.")

        while self.display_running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.display_running = False
                    break
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    # Check if the button is clicked
                    if button_rect.collidepoint(event.pos):
                        save_calibration_image(self)

            # Draw the video frame
            if hasattr(self, "cur_frame") and self.cur_frame is not None:
                frame_surface = pygame.surfarray.make_surface(cv2.cvtColor(self.cur_frame, cv2.COLOR_BGR2RGB))
                self.screen.blit(pygame.transform.rotate(frame_surface, -90), (0, 0))

            # Draw the button
            mouse_pos = pygame.mouse.get_pos()
            if button_rect.collidepoint(mouse_pos):
                pygame.draw.rect(self.screen, button_hover_color, button_rect)
            else:
                pygame.draw.rect(self.screen, button_color, button_rect)

            # Draw button text
            text_surface = font.render("Take Calibration Image", True, text_color)
            text_rect = text_surface.get_rect(center=button_rect.center)
            self.screen.blit(text_surface, text_rect)

            pygame.display.flip()
            self.clock.tick(30)  # Limit to 30 FPS for smoother updates

        pygame.quit()
        self.logger.info("Pygame display loop exited.")

    # Launch the display loop in a separate thread
    self.display_thread = threading.Thread(target=display_loop)
    self.display_thread.start()


def save_calibration_image(self: "TelloDrone"):
    if hasattr(self, "cur_frame") and self.cur_frame is not None:
        img_path = f"calibrate/img/{self.init_time}/frame-{self.frame_idx}.jpg"
        cv2.imwrite(img_path, self.cur_frame)
        self.logger.info(f"Calibration image saved: {img_path}")
    else:
        self.logger.warning("No frame available to save.")


def process_frame(self: "TelloDrone", frame: av.VideoFrame):
    self.cur_frame = frame.to_ndarray(format="bgr24")  # Save the frame for display
    self.video_writer.write(self.cur_frame)

    # Start a thread for the active video task if the condition is met
    if self.frame_idx >= 100 and self.active_vid_task:
        if self.active_vid_task_thread is None or not self.active_vid_task_thread.is_alive():
            def task_wrapper():
                try:
                    self.active_vid_task(self.cur_frame)
                except Exception as e:
                    self.logger.error(f"Error in active video task: {e}")
                finally:
                    self.active_vid_task_thread = None

            # Create and start a new thread for the task
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
                
                self.process_frame(frame)
                
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