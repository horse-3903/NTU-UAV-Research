import pybullet
import pybullet_data

import json
import time
import random

import traceback

from apf import apf, apf_with_bounds, apf_with_smooth_decay
from vector import Vector3D

class TelloDroneSim:
    def __init__(self):
        self.x_bounds = (-0.75, 6.85)
        self.y_bounds = (0, 4.5)
        self.z_bounds = (-4.25, 0.0)
        
        self.cur_pos = Vector3D(6.40, 2.35, -2.80)
        self.target_pos = Vector3D(-0.20, 2.15, -1.85)
        self.cur_orient = pybullet.getQuaternionFromEuler([1.57079632679, 0, 3.1415926536])
        
        self.obstacles = []
        self.pos = []
        
        self.attract_coeff = 100
        self.repel_coeff = 10
        self.influence_dist = 1.5
        self.bounds_influence_dist = 0.5
        self.decay_factor = 2
        self.bounds_decay_factor = 1
        self.global_delta = (self.cur_pos - self.target_pos).magnitude()
        
        self.active_task = None
        
    def generate_obstacles(self, num=10, radius=0.75):
        self.obstacles = [(Vector3D(random.uniform(*self.x_bounds), random.uniform(*self.y_bounds), random.uniform(*self.z_bounds)), radius) for _ in range(num)]
    
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
    
    def run_apf(self):
        total_force, heading_angle, attract_force, repel_force = apf_with_smooth_decay(
            cur_pos=self.cur_pos, 
            target_pos=self.target_pos, 
            obstacles=self.obstacles, 
            attract_coeff=self.attract_coeff, 
            repel_coeff=self.repel_coeff, 
            decay_factor=self.decay_factor,
            x_bounds=self.x_bounds,
            y_bounds=self.y_bounds,
            z_bounds=self.z_bounds,
            bounds_decay_factor=self.bounds_decay_factor
        )
                
        print("Attraction Force :", attract_force)
        print("Repulsion Force :", repel_force)
        print("Resultant Force :", total_force)

        scalar = 1
        velocity_x = total_force.x / self.local_delta * scalar
        velocity_y = total_force.y / self.local_delta * scalar
        velocity_z = total_force.z / self.local_delta * scalar

        print("Control Velocity :", (velocity_x, velocity_y, velocity_z))
        
        print()
        
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
        
        # Draw translucent boundary walls
        boundary_color = [255, 255, 255, 0.25]  # RGBA with alpha for translucency

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
                baseCollisionShapeIndex=collision_shape_id,
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