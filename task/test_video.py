import time
from functools import partial
from tellodrone import TelloDrone

tello = TelloDrone()

def main():
    tello.startup_video()
    # tello.active_img_task = partial(tello.run_depth_model, manual=True)
    time.sleep(1000)
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