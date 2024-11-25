import av
import cv2

import threading

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tellodrone.core import TelloDrone

def process_frame(self: "TelloDrone", frame: av.VideoFrame):
    img = frame.to_ndarray(format='bgr24')                    
    self.video_writer.write(img)
    
    if self.frame_idx >= 100 and self.active_vid_task:
        self.active_vid_task(img)

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

    if not self.video_writer:
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
    self.video_thread.join()
    self.video_writer.release()
    self.logger.info("Video processing thread stopped, and video file saved.")