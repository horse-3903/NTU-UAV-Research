import time
from functools import partial
from tellodrone import TelloDrone

tello = TelloDrone()

def main():
    tello.active_img_task = partial(tello.run_depth_model, manual=True)
    tello.run_objective(display=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        for _ in range(1000):
            tello.logger.critical("catch 1")
        tello.shutdown(error=True, reason=e)