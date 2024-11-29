import time

import signal

import logging

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
    tello.logger.critical("Ctrl+C Keyboard Interrupt")
    tello.shutdown(error=True, reason="Ctrl+C Keyboard Interrupt")

def main():    
    signal.signal(signal.SIGINT, signal_handler)
    
    rospy.init_node('drone_subscriber', anonymous=True)

    rospy.Subscriber('/nlink_linktrack_nodeframe1', LinktrackNodeframe1, linktrack_callback, queue_size=1)

    tello.set_target_pos(Vector3D(0.30, 1.75, -1.40))
    # tello.set_target_pos(Vector3D(5.95, 2.30, -0.85))
    
    # tello.add_obstacle((Vector3D(3.6054744958877567, 2.1556083091614027, -2.720276257489722), 0.5385408378125782))
    
    tello.run_objective(display=True)

    rospy.spin()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        tello.shutdown(error=True, reason=e)