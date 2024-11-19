#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped

# Callback function to handle the incoming PoseStamped messages
def pose_callback(msg:PoseStamped):
    # Extract position data from the message
    x = msg.pose.position.x
    y = msg.pose.position.y
    z = msg.pose.position.z

    # Print the received position
    rospy.loginfo(f"Drone Position - x: {x}, y: {y}, z: {z}")

def main():
    # Initialize the ROS node
    rospy.init_node('drone_pose_subscriber', anonymous=True)

    # Subscribe to the drone's PoseStamped topic
    rospy.Subscriber('/vrpn_client_node/Drone_HCI/pose', PoseStamped, pose_callback)

    # Keep the node running
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
