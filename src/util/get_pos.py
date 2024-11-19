from datetime import datetime

import rospy
from nlink_parser.msg import LinktrackNodeframe1, LinktrackNode1

file = f"log/log-{datetime.now().strftime('%d-%m-%Y_%H:%M:%S')}.txt"

def linktrack_callback(msg: LinktrackNodeframe1):
    node: LinktrackNode1 = msg.nodes[0]
    pos = node.pos_3d
    pos = map(str, pos)
    with open(file, "a") as f:
        f.write(" ".join(pos) + "\n")
    
def main():
    with open(file, "w+") as f:
        f.write("")
    
    rospy.init_node('drone_pose_subscriber', anonymous=True)

    rospy.Subscriber('/nlink_linktrack_nodeframe1', LinktrackNodeframe1, linktrack_callback, queue_size=1)

    rospy.spin()
    
if __name__ == "__main__":
    main()