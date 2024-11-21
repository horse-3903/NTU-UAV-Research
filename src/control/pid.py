from task.tellodrone.vector import Vector3D
from simple_pid import PID

current_position = Vector3D(0, 0, 0)
waypoint = Vector3D(1000, 5000, 2000)

pid_x = PID(Kp=1.0, Ki=0.1, Kd=0.05, setpoint=waypoint.x, output_limits=(0, 10))
pid_y = PID(Kp=1.0, Ki=0.1, Kd=0.05, setpoint=waypoint.y, output_limits=(0, 10))
pid_z = PID(Kp=1.0, Ki=0.1, Kd=0.05, setpoint=waypoint.z, output_limits=(0, 10))

for _ in range(100):    

    control_x = pid_x(current_position.x)
    control_y = pid_y(current_position.y)
    control_z = pid_z(current_position.z)
    

    current_position.x += control_x
    current_position.y += control_y
    current_position.z += control_z
    
    error = current_position - waypoint
    
    print(f"Current Position: {current_position}")
    print(f"Target position: {waypoint}")
    print(f"Error Magnitude : {error.magnitude()}")
    print()

    if error.magnitude() < 0.1:
        print("Waypoint reached!")
        break
