import logging

import signal

from tellodrone import TelloDrone

import rospy
from nlink_parser.msg import LinktrackNode1, LinktrackNodeframe1

from task.vector import Vector3D

tello = TelloDrone()

def linktrack_callback(data):
    node: LinktrackNode1 = data.nodes[0]
    pos_arr = node.pos_3d

    tello.task_handler(pos_arr=pos_arr)

def signal_handler(sig, frame):
    logging.critical("Ctrl+C Keyboard Interrupt")
    tello.shutdown(error=True, reason="Ctrl+C Keyboard Interrupt")

def main():    
    signal.signal(signal.SIGINT, signal_handler)
    
    rospy.init_node('drone_subscriber', anonymous=True)

    rospy.Subscriber('/nlink_linktrack_nodeframe1', LinktrackNodeframe1, linktrack_callback, queue_size=1)

    tello.set_target_pos(Vector3D(-0.20, 2.0, -0.85))
    tello.set_obstacles([(Vector3D(4.0, 2.5, -2.75), 0.65)])
    tello.set_obstacles([(Vector3D(4.0, 2.5, -2.25), 0.65)])
    tello.set_obstacles([(Vector3D(1.7, 1.6, -2.75), 0.65)])
    tello.set_obstacles([(Vector3D(1.7, 1.6, -2.25), 0.65)])
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