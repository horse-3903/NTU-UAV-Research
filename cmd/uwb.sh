#!/bin/bash

# Set the parent directory path
PARENT_DIR="/home/horse3903/catkin_ws"

# Run catkin_make to build the workspace
# (cd "$PARENT_DIR" && catkin_make)

# Set the environment variable for software rendering
export LIBGL_ALWAYS_SOFTWARE=1

# Define a function to handle the trap
cleanup() {
    rosnode kill -a &
    exit 0
}

# Set trap to catch SIGINT (Ctrl+C)
trap cleanup SIGINT

# Check if roscore is running
if ! pgrep -x "roscore" > /dev/null; then
  # If roscore is not running, start roscore in the background
  roscore &  
  sleep 2  # Wait for roscore to start
fi

# Run your code with proper background handling
roslaunch nlink_parser linktrack.launch

# Wait for background jobs to finish and handle SIGINT properly
wait -n  # Wait for the first background job to finish
cleanup   # Run the cleanup function after any job finishes