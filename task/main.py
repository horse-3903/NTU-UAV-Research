from drone import TelloDrone

import rospy
from nlink_parser.msg import LinktrackNode1, LinktrackNodeframe1

tello = TelloDrone()

def linktrack_callback(data):
    # to assess node control (drone vs target) later
    node: LinktrackNode1 = data.nodes[0]
    pos_arr = node.pos_3d

    tello.task_handler(pos_arr=pos_arr)

def main():
    rospy.init_node('drone_subscriber', anonymous=True)

    rospy.Subscriber('/nlink_linktrack_nodeframe1', LinktrackNodeframe1, linktrack_callback, queue_size=20)

    rospy.spin()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
    finally:
        tello.shutdown()