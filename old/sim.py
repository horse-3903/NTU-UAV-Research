import os
import json

import time
import pybullet
import pybullet_data

from apf import apf

import random

import traceback

from vector import Vector3D

def test_apf():
    x_bounds = (-0.75, 6.85)
    y_bounds = (0, 4.5)
    z_bounds = (-4.25, 0.0)
    
    client = pybullet.connect(pybullet.GUI)
    if client < 0:
        raise RuntimeError("Failed to connect to PyBullet physics server.")
    
    pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())

    cur_pos = Vector3D(6.40, 2.35, -2.80)
    pybullet.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=90, cameraPitch=-30, cameraTargetPosition=cur_pos.to_arr())
    
    plane_id = pybullet.loadURDF("plane.urdf", basePosition=(5.0, 5.0, -2.80), globalScaling=2)
    drone = pybullet.loadURDF("duck_vhacd.urdf", basePosition=cur_pos.to_arr(), baseOrientation=pybullet.getQuaternionFromEuler([1.57079632679, 0, 3.1415926536]), useFixedBase=False, globalScaling=5)
    
    target_pos = Vector3D(-0.20, 2.15, -1.85)
    target_id = pybullet.loadURDF("soccerball.urdf", basePosition=target_pos.to_arr(), useFixedBase=True, globalScaling=0.5)
    
    obstacles = [(Vector3D(random.uniform(*x_bounds), random.uniform(*y_bounds), random.uniform(*z_bounds)), 0.75) for _ in range(10)]
    for ob in obstacles:
        pybullet.loadURDF("sphere2.urdf", basePosition=ob[0].to_arr(), useFixedBase=True, globalScaling=ob[1])

    attract_coeff = 50
    repul_coeff = 20
    global_delta = (cur_pos - target_pos).magnitude()

    while True:
        try:
            if not pybullet.isConnected(client):
                print("Disconnected from PyBullet physics server.")
                break
            
            cur_pos, cur_orient = pybullet.getBasePositionAndOrientation(drone, physicsClientId=client)
            cur_pos = Vector3D.from_arr(cur_pos)
            local_delta = (cur_pos - target_pos).magnitude()
            
            if local_delta <= 0.3:
                print("Success: Drone reached the target!")
                break
            
            print("Current Position :", cur_pos)
            print("Target Position :", target_pos)
            print("Local Delta :", )
    
            attract_force, repel_force, total_force = apf(
                current_pos=cur_pos, 
                target_pos=target_pos, 
                obstacles=obstacles,
                attraction_coeff_base=attract_coeff, 
                repulsion_coeff=repul_coeff, 
                normalise_val=global_delta
            )
            
            print("Attraction Force :", attract_force)
            print("Repulsion Force :", repel_force)
            print("Resultant Force :", total_force)

            scalar = 0.25
            velocity_x = total_force.x / local_delta * scalar
            velocity_y = total_force.y / local_delta * scalar
            velocity_z = total_force.z / local_delta * scalar

            print("Control Velocity :", (velocity_x, velocity_y, velocity_z))
            
            print()
            
            pybullet.resetBaseVelocity(
                drone, (velocity_x, velocity_y, velocity_z), (0, 0, 0)
            )

            pybullet.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=90, cameraPitch=-30, cameraTargetPosition=cur_pos.to_arr())
            
            pybullet.stepSimulation()
            time.sleep(0.05)
        except Exception as e:
            print(traceback.format_exc())
            break
    pybullet.disconnect()
    
def test_log(log_dir):
    log_config_dir = f"{log_dir}/log-config.json"
    with open(log_config_dir, "r") as f:
        config = json.load(f)
    
    log_file = f"{log_dir}/log-pos.log"

    with open(log_file, "r") as f:
        data = f.read().splitlines()
        pos = [Vector3D.from_arr(list(map(float, line.split()[3:]))) for line in data]
    
    client = pybullet.connect(pybullet.GUI)
    if client < 0:
        raise RuntimeError("Failed to connect to PyBullet physics server.")
    
    pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())

    cur_pos = Vector3D.from_arr(config["takeoff_pos"])
    pybullet.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=90, cameraPitch=-30, cameraTargetPosition=cur_pos.to_arr())
    
    plane_id = pybullet.loadURDF("plane.urdf", basePosition=config["takeoff_pos"], globalScaling=2)
    drone = pybullet.loadURDF("duck_vhacd.urdf", basePosition=cur_pos.to_arr(), baseOrientation=pybullet.getQuaternionFromEuler([1.57079632679, 0, 3.1415926536]), useFixedBase=False, globalScaling=5)
    
    target_pos = Vector3D.from_arr(config["target_pos"])
    target = pybullet.loadURDF("soccerball.urdf", basePosition=target_pos.to_arr(), useFixedBase=True, globalScaling=0.5)
    
    obstacles = config["obstacles"]
    for ob in obstacles:
        pybullet.loadURDF("sphere2.urdf", basePosition=ob[0], useFixedBase=True, globalScaling=ob[1])
    
    for cur_pos in pos:
        try:
            if not pybullet.isConnected(client):
                print("Disconnected from PyBullet physics server.")
                break
            
            _, cur_orient = pybullet.getBasePositionAndOrientation(drone, physicsClientId=client)
            pybullet.resetBasePositionAndOrientation(
                drone, cur_pos.to_arr(), cur_orient
            )

            local_delta = (cur_pos - target_pos).magnitude()
            if local_delta <= 0.3:
                print("Success: Drone reached the target!")
                pybullet.disconnect()
                return log_dir

            pybullet.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=90, cameraPitch=-30, cameraTargetPosition=cur_pos.to_arr())
            pybullet.stepSimulation()
            time.sleep(0.025)
        except Exception as e:
            print(traceback.format_exc())
            break
    pybullet.disconnect()

if __name__ == "__main__":
    success_lst = []
    logs = sorted(os.listdir("logs"))
    
    for i in range(len(logs)):
        res = test_log("logs/"+logs[i]+"/")
        if res:
            success_lst.append(res)
            
    print(success_lst)
    
    # test_apf()