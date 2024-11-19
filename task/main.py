import sys
from tellopy import Tello

from task.depth_model import estimate_depth

drone = Tello()

def shutdown():
    drone.land()
    drone.quit()
    sys.exit(0)
    
def handle_movement():
    pass

def main():
    pass

if __name__ == "__main__":
    try:
        main()
    except:
        shutdown()