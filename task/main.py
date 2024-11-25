import logging

import signal

from tellodrone import TelloDrone

import rospy
from nlink_parser.msg import LinktrackNodeframe1

from vector import Vector3D

tello = TelloDrone()

def linktrack_callback(data: LinktrackNodeframe1):
    node = data.nodes[0]
    pos_arr = node.pos_3d

    tello.task_handler(pos_arr=pos_arr)

def signal_handler(sig, frame):
    logging.critical("Ctrl+C Keyboard Interrupt")
    tello.shutdown(error=True, reason="Ctrl+C Keyboard Interrupt")

def main():    
    signal.signal(signal.SIGINT, signal_handler)
    
    rospy.init_node('drone_subscriber', anonymous=True)

    rospy.Subscriber('/nlink_linktrack_nodeframe1', LinktrackNodeframe1, linktrack_callback, queue_size=1)

    tello.set_target_pos(Vector3D(-0.30, 1.75, -1.40))
    tello.add_obstacle((Vector3D(2.35, 2.05, -2.85), 0.75))
    tello.add_obstacle((Vector3D(2.35, 2.05, -2.35), 0.75))
    tello.run_objective()

    rospy.spin()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        tello.shutdown(error=True, reason=e)
    else:
        tello.shutdown(error=False)
    finally:
        tello.shutdown(error=False)