import sys
import time

import rospy
from nlink_parser.msg import LinktrackNode1, LinktrackNodeframe1

from tellopy import Tello

from vector import Vector3D

class TelloDrone:
    def __init__(self) -> None:
        self.pos = Vector3D(0, 0, 0)
        self.target = Vector3D(0, 0, 0)
        
        self.task_queue = []
        self.active_task = function
        
        self.drone = Tello()
        self.running = False
        
    def set_target(self, target: Vector3D) -> None:
        self.target = target
        
    def startup(self) -> None:
        if self.target.is_origin():
            return
        
        self.running = True
        
        self.drone.connect()
        self.drone.wait_for_connection(10)
        
        self.drone.takeoff()

        time.sleep(2)
        
    def shutdown(self) -> None:
        self.running = False
        
        self.drone.backward(0)
        self.drone.land()
        self.drone.quit()
        
        rospy.signal_shutdown()
        
        sys.exit(0)
        
    def check_bounds(self, x_bounds, y_bounds, z_bounds) -> None:
        within_bounds = (x_bounds[0] <= self.x <= x_bounds[1] and  y_bounds[0] <= self.y <= y_bounds[1] and z_bounds[0] <= self.z <= z_bounds[1])
            
        if not within_bounds:
            self.shutdown()
    
    def task_handler(self, pos_arr) -> None:
        self.pos = Vector3D.from_arr(pos_arr)
        
        self.check_bounds()
        self.active_task()
        
    def plan_path() -> None:
        pass
        
tello = TelloDrone()

def linktrack_callback(data):
    node: LinktrackNode1 = data.nodes[0]
    pos_arr = node.pos_3d

    tello.task_handler(pos_arr=pos_arr)

def main():
    rospy.init_node('drone_subscriber', anonymous=True)

    rospy.Subscriber('/nlink_linktrack_nodeframe1', LinktrackNodeframe1, linktrack_callback, queue_size=20)

    # Keep the node running
    rospy.spin()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
    finally:
        tello.shutdown()