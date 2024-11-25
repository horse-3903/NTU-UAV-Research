import time
from tellodrone import TelloDrone

tello = TelloDrone()

def main():
    tello.active_vid_task = tello.run_depth_model
    tello.load_depth_model()
    tello.startup_video()
    time.sleep(500)
    tello.shutdown(error=False)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        tello.shutdown(error=True, reason=e)
    else:
        tello.shutdown(error=False)
    finally:
        tello.shutdown(error=False)