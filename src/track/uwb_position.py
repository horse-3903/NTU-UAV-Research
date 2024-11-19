#!/usr/bin/env python3

import rospy
from nlink_parser.msg import LinktrackNodeframe1

# Callback function to handle the incoming PoseStamped messages
def linktrack_callback(msg: LinktrackNodeframe1):
    # Extract position data from the message
    node = msg.nodes[0]
    pos = node.pos_3d

    rospy.loginfo(f"Drone Position - x: {pos.x}, y: {pos.y}, z: {pos.z}")

def main():
    # Initialize the ROS node
    rospy.init_node('drone_pose_subscriber', anonymous=True)

    # Subscribe to the drone's PoseStamped topic
    rospy.Subscriber('/nlink_linktrack_nodeframe1', LinktrackNodeframe1, linktrack_callback, queue_size=1)

    # Keep the node running
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
