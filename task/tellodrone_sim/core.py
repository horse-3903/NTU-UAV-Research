import pybullet
import pybullet_data

import json
import time
import random

import traceback

import math

from simple_pid import PID

from apf import apf, apf_with_bounds

from vector import Vector3D

import heapq

class TelloDroneSim:
    def __init__(self):
        self.x_bounds = (-0.75, 6.85)
        self.y_bounds = (0, 4.5)
        self.z_bounds = (-4.25, -1.25)
        
        self.start_pos = Vector3D(6.40, 2.35, -2.80)
        self.cur_pos = Vector3D(6.40, 2.35, -2.80)
        self.target_pos = Vector3D(-0.20, 2.15, -4.00)
        
        self.cur_orient = pybullet.getQuaternionFromEuler([1.57079632679, 0, 3.1415926536])
        
        self.cur_idx = 0
        
        self.obstacles = []
        self.path = []
        self.pos = []
        
        self.attract_coeff = 30
        self.repel_coeff = 10
        self.influence_dist = 0.5
        self.bounds_influence_dist = 0.5
        
        self.global_delta = (self.cur_pos - self.target_pos).magnitude()
        
        self.active_task = None
        
    def generate_obstacles(self, num=10, radius=0.75):
        self.obstacles = [(Vector3D(random.uniform(*self.x_bounds), random.uniform(*self.y_bounds), random.uniform(*self.z_bounds)), radius) for _ in range(num)]
        
    def plan_sine_path(self, num: int, amplitude: float = 1.0, frequency: float = 1.0):
        if num < 2:
            raise ValueError("The number of waypoints must be at least 2.")
        
        delta = (self.target_pos - self.start_pos)
        step_vector = delta / (num - 1)
        
        self.path = []
        for i in range(num):
            t = i / (num - 1)
            base_point = self.start_pos + step_vector * i

            sine_offset = amplitude * math.sin(2 * math.pi * frequency * t)
            self.path.append(Vector3D(base_point.x, base_point.y + sine_offset, base_point.z))
    
    def load_config(self, log_dir=None):
        log_config_dir = f"{log_dir}/log-config.json"
        with open(log_config_dir, "r") as f:
            config = json.load(f)

        log_file = f"{log_dir}/log-pos.log"

        with open(log_file, "r") as f:
            data = f.read().splitlines()
            self.pos = [Vector3D.from_arr(list(map(float, line.split()[3:]))) for line in data]
        
        self.pos_idx = 0
        self.cur_pos = Vector3D.from_arr(config["takeoff_pos"])
        self.target_pos = Vector3D.from_arr(config["target_pos"])
        self.obstacles = [(Vector3D.from_arr(ob[0]), ob[1]) for ob in config["obstacles"]]
            
    def run_log(self):
        self.cur_pos = self.pos[self.pos_idx]
        self.set_position(self.cur_pos)
        
        self.pos_idx += 1
        
    def set_position(self, pos: Vector3D):
        pybullet.resetBasePositionAndOrientation(self.drone_id, pos.to_arr(), self.cur_orient)
        
    def set_velocity(self, vel: Vector3D):
        pybullet.resetBaseVelocity(self.drone_id, vel.to_arr(), (0, 0, 0))
    
    
    def a_star_waypoints(self, grid_resolution=0.5):
        # Convert bounds to grid points
        x_steps = int((self.x_bounds[1] - self.x_bounds[0]) / grid_resolution) + 1
        y_steps = int((self.y_bounds[1] - self.y_bounds[0]) / grid_resolution) + 1
        z_steps = int((self.z_bounds[1] - self.z_bounds[0]) / grid_resolution) + 1

        # Helper to discretize a point to the grid
        def to_grid(pos):
            return (
                int((pos.x - self.x_bounds[0]) / grid_resolution),
                int((pos.y - self.y_bounds[0]) / grid_resolution),
                int((pos.z - self.z_bounds[0]) / grid_resolution)
            )

        # Helper to convert grid coordinates back to world space
        def to_world(grid_pos):
            return Vector3D(
                self.x_bounds[0] + grid_pos[0] * grid_resolution,
                self.y_bounds[0] + grid_pos[1] * grid_resolution,
                self.z_bounds[0] + grid_pos[2] * grid_resolution,
            )

        # A* priority queue and costs
        open_set = []
        heapq.heappush(open_set, (0, to_grid(self.start_pos)))
        came_from = {}
        g_score = {to_grid(self.start_pos): 0}
        f_score = {to_grid(self.start_pos): (self.start_pos - self.target_pos).magnitude()}

        # Directions for movement in the grid (26 directions in 3D space)
        directions = [
            (dx, dy, dz)
            for dx in [-1, 0, 1]
            for dy in [-1, 0, 1]
            for dz in [-1, 0, 1]
            if not (dx == 0 and dy == 0 and dz == 0)
        ]

        # Obstacles as grid positions
        obstacle_grid = {to_grid(ob[0]) for ob in self.obstacles}

        target_grid = to_grid(self.target_pos)

        while open_set:
            _, current = heapq.heappop(open_set)

            # Check if target is reached
            if current == target_grid:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(to_world(current))
                    current = came_from[current]
                path.append(self.start_pos)  # Add the start position
                self.path = path[::-1]  # Reverse path
                return

            for direction in directions:
                neighbor = (
                    current[0] + direction[0],
                    current[1] + direction[1],
                    current[2] + direction[2],
                )

                # Check if the neighbor is within bounds
                if not (0 <= neighbor[0] < x_steps and 0 <= neighbor[1] < y_steps and 0 <= neighbor[2] < z_steps):
                    continue

                # Check if the neighbor is an obstacle
                if neighbor in obstacle_grid:
                    continue

                # Calculate tentative g_score
                tentative_g_score = g_score[current] + math.sqrt(
                    direction[0]**2 + direction[1]**2 + direction[2]**2
                )

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + (to_world(neighbor) - self.target_pos).magnitude()
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        raise ValueError("No path found to the target.")

    
    def run_apf(self):
        # Check if all waypoints are reached
        if self.cur_idx >= len(self.path):
            print("All waypoints reached!")
            return

        # Set the current waypoint as the temporary target
        waypoint = self.path[self.cur_idx]
        local_delta = (self.cur_pos - waypoint).magnitude()

        # If close enough to the waypoint, move to the next
        if local_delta <= 0.05:  # Threshold for reaching the waypoint
            print(f"Waypoint {self.cur_idx} reached at position: {self.cur_pos}")
            self.cur_idx += 1
            if self.cur_idx >= len(self.path):
                print("All waypoints reached.")
                return
            waypoint = self.path[self.cur_idx]
            print(f"Moving to next waypoint: {waypoint}")

        # Calculate forces using APF
        total_force, attract_force, repel_force = apf_with_bounds(
            cur_pos=self.cur_pos,
            target_pos=waypoint,
            obstacles=self.obstacles,
            attract_coeff=self.attract_coeff,
            repel_coeff=self.repel_coeff,
            influence_dist=self.influence_dist,
            x_bounds=self.x_bounds,
            y_bounds=self.y_bounds,
            z_bounds=self.z_bounds,
            bounds_influence_dist=self.bounds_influence_dist
        )

        # Debugging info
        print("Attraction Force :", attract_force)
        print("Repulsion Force :", repel_force)
        print("Resultant Force :", total_force)

        # Compute velocity for the drone
        local_delta = max(local_delta, 0.1)  # Avoid division by zero
        scalar = 1.0  # Tuning factor for velocity
        velocity_x = total_force.x / local_delta * scalar
        velocity_y = total_force.y / local_delta * scalar
        velocity_z = total_force.z / local_delta * scalar

        print("Control Velocity :", (velocity_x, velocity_y, velocity_z))
        print()

        # Set the drone velocity
        self.set_velocity(Vector3D(velocity_x, velocity_y, velocity_z))

    
    def start_sim(self):
        self.client_id = pybullet.connect(pybullet.GUI)
        if self.client_id < 0:
            raise RuntimeError("Failed to connect to PyBullet physics server.")
        
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())

        pybullet.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=90, cameraPitch=-30, cameraTargetPosition=self.cur_pos.to_arr())
        
        # Load environment and drone
        # self.plane_id = pybullet.loadURDF("plane.urdf", basePosition=(5.0, 5.0, -2.80), globalScaling=2)
        self.drone_id = pybullet.loadURDF("duck_vhacd.urdf", basePosition=self.cur_pos.to_arr(), baseOrientation=pybullet.getQuaternionFromEuler([1.57079632679, 0, 3.1415926536]), useFixedBase=False, globalScaling=5)
        self.target_id = pybullet.loadURDF("soccerball.urdf", basePosition=self.target_pos.to_arr(), useFixedBase=True, globalScaling=0.5)

        for ob in self.obstacles:
            pybullet.loadURDF("sphere2.urdf", basePosition=ob[0].to_arr(), useFixedBase=True, globalScaling=ob[1])
            
        for waypoint in self.path:
            visual_shape_id = pybullet.createVisualShape(
                shapeType=pybullet.GEOM_SPHERE,
                radius=0.1,
                rgbaColor=[1, 0, 0, 0.5]
            )
            pybullet.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=visual_shape_id,
                basePosition=waypoint.to_arr()
            )
        
        # Draw translucent boundary walls
        boundary_color = [255, 255, 255, 0.15]  # RGBA with alpha for translucency

        def create_wall(center, size):
            visual_shape_id = pybullet.createVisualShape(
                shapeType=pybullet.GEOM_BOX,
                halfExtents=size,
                rgbaColor=boundary_color
            )
            collision_shape_id = pybullet.createCollisionShape(
                shapeType=pybullet.GEOM_BOX,
                halfExtents=size
            )
            pybullet.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=visual_shape_id,
                basePosition=center
            )

        # Walls along x-bounds (parallel to y-axis)
        create_wall(center=((self.x_bounds[0] + self.x_bounds[1]) / 2, self.y_bounds[0], (self.z_bounds[0] + self.z_bounds[1]) / 2),
                    size=[(self.x_bounds[1] - self.x_bounds[0]) / 2, 0.1, (self.z_bounds[1] - self.z_bounds[0]) / 2])
        create_wall(center=((self.x_bounds[0] + self.x_bounds[1]) / 2, self.y_bounds[1], (self.z_bounds[0] + self.z_bounds[1]) / 2),
                    size=[(self.x_bounds[1] - self.x_bounds[0]) / 2, 0.1, (self.z_bounds[1] - self.z_bounds[0]) / 2])

        # Walls along y-bounds (parallel to x-axis)
        create_wall(center=(self.x_bounds[0], (self.y_bounds[0] + self.y_bounds[1]) / 2, (self.z_bounds[0] + self.z_bounds[1]) / 2),
                    size=[0.1, (self.y_bounds[1] - self.y_bounds[0]) / 2, (self.z_bounds[1] - self.z_bounds[0]) / 2])
        create_wall(center=(self.x_bounds[1], (self.y_bounds[0] + self.y_bounds[1]) / 2, (self.z_bounds[0] + self.z_bounds[1]) / 2),
                    size=[0.1, (self.y_bounds[1] - self.y_bounds[0]) / 2, (self.z_bounds[1] - self.z_bounds[0]) / 2])

        # Floor and ceiling (z-bounds)
        create_wall(center=((self.x_bounds[0] + self.x_bounds[1]) / 2, (self.y_bounds[0] + self.y_bounds[1]) / 2, self.z_bounds[0]),
                    size=[(self.x_bounds[1] - self.x_bounds[0]) / 2, (self.y_bounds[1] - self.y_bounds[0]) / 2, 0.1])
        create_wall(center=((self.x_bounds[0] + self.x_bounds[1]) / 2, (self.y_bounds[0] + self.y_bounds[1]) / 2, self.z_bounds[1]),
                    size=[(self.x_bounds[1] - self.x_bounds[0]) / 2, (self.y_bounds[1] - self.y_bounds[0]) / 2, 0.1])

            
    def run_sim(self):
        while True:
            try:
                if not pybullet.isConnected(self.client_id):
                    print("Disconnected from PyBullet physics server.")
                    break
                
                self.cur_pos, cur_orient = pybullet.getBasePositionAndOrientation(self.drone_id, physicsClientId=self.client_id)
                self.cur_pos = Vector3D.from_arr(self.cur_pos)
                self.local_delta = (self.cur_pos - self.target_pos).magnitude()
                
                if self.local_delta <= 0.6:
                    print("Success: Drone reached the target!")
                    break
                
                print("Current Position :", self.cur_pos)
                print("Target Position :", self.target_pos)
                print("Local Delta :", self.local_delta)

                if self.active_task:
                    self.active_task()
                    pybullet.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=90, cameraPitch=-30, cameraTargetPosition=self.cur_pos.to_arr())
                
                pybullet.stepSimulation()
                time.sleep(0.05)
            except Exception as e:
                print(traceback.format_exc())
                break
            
        pybullet.disconnect()