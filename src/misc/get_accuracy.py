import sys

import time
from datetime import datetime

from tellopy import Tello

import rospy
from nlink_parser.msg import LinktrackNodeframe1, LinktrackNode1

# Setup log file with current date and time
log_file = f"logs/log-{datetime.now().strftime('%d-%m-%Y_%H:%M:%S')}.txt"
drone = Tello()

# Initialize time and position variables
start_time = None
start_pos = []
final_pos = []

# Function for Shutdown
def shutdown():
    drone.backward(0)
    time.sleep(1)
    
    drone.land()
    time.sleep(1)
    
    drone.quit()
    rospy.signal_shutdown("Exit")
    sys.exit(0)

# Callback to handle flight data events
def flight_data_callback(event, sender, data):
    rospy.loginfo(data)

# Function to handle the drone's movement logic
def handle_drone_movement(pos):
    global final_pos

    # Check if the final position is not yet determined
    if not final_pos:
        return

    # Control the drone to move forward until it reaches the final position
    if pos[0] > final_pos[0]:
        drone.forward(100)
        time.sleep(0.1)
    
    # if pos[1] < final_pos[1]:
    #     drone.right(7)
    #     time.sleep(0.1)
    
    # if pos[1] > final_pos[1]:
    #     drone.left(7)
    #     time.sleep(0.1)
    
    if pos[0] <= final_pos[0]:
        # Stop the drone and land
        shutdown()

# Callback to handle node position messages
def linktrack_callback(msg: LinktrackNodeframe1):
    global start_pos, final_pos
    
    # Get current time and compute elapsed time
    cur_time = rospy.get_time()
    delta_time = cur_time - start_time

    # Extract position from message
    node: LinktrackNode1 = msg.nodes[0]
    pos = node.pos_3d

    # Log position data
    with open(log_file, "a") as f:
        f.write(str(delta_time) + " " + " ".join(map(str, pos)) + "\n")

    # Calculate start and final positions
    if delta_time <= 5:
        if not start_pos:
            start_pos = pos
        else:
            # Averaging the start position
            start_pos = [
                (start_pos[0] + pos[0]) / 2,
                (start_pos[1] + pos[1]) / 2,
                (start_pos[2] + pos[2]) / 2,
            ]
    else:
        if not final_pos:
            # Offset the final position by -1 on the x-axis
            final_pos = [
                (start_pos[0] + pos[0]) / 2 - 1,
                (start_pos[1] + pos[1]) / 2,
                (start_pos[2] + pos[2]) / 2,
            ]

    # Call the separate movement handling function
    handle_drone_movement(pos)

def main():
    global start_time

    # Create or clear the log file
    with open(log_file, "w") as f:
        f.write("")

    # Initialize the ROS node
    rospy.init_node('drone_pose_subscriber', anonymous=True)
    
    # Record the start time
    start_time = rospy.get_time()

    # Subscribe to the position topic
    rospy.Subscriber('/nlink_linktrack_nodeframe1', LinktrackNodeframe1, linktrack_callback, queue_size=15)

    # Connect to the drone and wait for the connection
    drone.connect()
    drone.wait_for_connection(10)
    
    drone.subscribe(drone.EVENT_FLIGHT_DATA, flight_data_callback)

    # Delay to ensure the connection is stable
    time.sleep(2)

    # Takeoff and start the ROS event loop
    drone.takeoff()
    drone.down(10)
    time.sleep(1)
    
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Ensure the drone lands and disconnects
        shutdown()