import av
import tellopy
import pygame
import numpy as np
import cv2

def main():
    # Initialize the Tello drone
    drone = tellopy.Tello()

    try:
        # Connect and start video stream
        drone.connect()
        drone.wait_for_connection(60.0)
        drone.start_video()
        
        container = None
        
        # Wait until the video stream is ready
        while container is None:
            container = av.open(drone.get_video_stream())

        # Initialize Pygame
        pygame.init()
        screen = pygame.display.set_mode((960, 720))
        pygame.display.set_caption("Tello Video Stream")

        frame_count = 0
        running = True

        while running:
            try:
                # Decode video frames
                for frame in container.decode(video=0):
                    frame_count += 1

                    img = frame.to_ndarray(format='bgr24')
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    surface = pygame.surfarray.make_surface(np.rot90(img))
                    screen.blit(surface, (0, 0))
                    pygame.display.update()

                    # Handle Pygame events
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                            break

                    if not running:
                        break

            except Exception as e:
                print(f"Error decoding frame: {e}")
                break

    except Exception as e:
        print(f"Error: {e}")

    finally:
        # Land the drone and clean up resources
        drone.land()
        drone.quit()
        pygame.quit()

if __name__ == "__main__":
    main()
