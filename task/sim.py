import os
import json

import time
import pybullet
import pybullet_data

from typing import List, Tuple
import math

import random

import traceback

def vector_subtract(v1: Tuple[float, float, float], v2: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2])

def vector_add(v1: Tuple[float, float, float], v2: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2])

def vector_magnitude(v: Tuple[float, float, float]) -> float:
    return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)

def vector_scale(v: Tuple[float, float, float], scale: float) -> Tuple[float, float, float]:
    return (v[0] * scale, v[1] * scale, v[2] * scale)

def vector_normalize(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    mag = vector_magnitude(v)
    if mag == 0:
        return (0.0, 0.0, 0.0)
    return vector_scale(v, 1.0 / mag)

def apf(
    current_pos: Tuple[float, float, float],
    target_pos: Tuple[float, float, float],
    obstacles: List[Tuple[Tuple[float, float, float], float]],
    attraction_coeff_base: float = 1.0,
    repulsion_coeff: float = 1.0,
    normalise_val: float = 10,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]:
    # Calculate the difference between target and current position
    delta = vector_subtract(target_pos, current_pos)
    distance_goal = vector_magnitude(delta)

    # Attractive force
    if distance_goal == 0:
        attractive_force = (0.0, 0.0, 0.0)
    else:
        attraction_coeff = attraction_coeff_base * (distance_goal / normalise_val)
        attractive_force = vector_scale(delta, attraction_coeff)

    # Repulsive force calculation
    repulsive_force = (0.0, 0.0, 0.0)
    
    for obs, radius in obstacles:
        delta_obs = vector_subtract(current_pos, obs)
        distance_obs = vector_magnitude(delta_obs) - radius

        if distance_obs > 0:
            repulsion_strength = repulsion_coeff / (distance_obs**2)
            repulsive_component = vector_scale(vector_normalize(delta_obs), repulsion_strength)
            repulsive_force = vector_add(repulsive_force, repulsive_component)

    total_force = vector_add(attractive_force, repulsive_force)

    return attractive_force, repulsive_force, total_force

def test_apf():
    x_bounds = (-0.75, 6.85)
    y_bounds = (0, 4.5)
    z_bounds = (-4.25, 0.0)
    
    client = pybullet.connect(pybullet.GUI)
    if client < 0:
        raise RuntimeError("Failed to connect to PyBullet physics server.")
    
    pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())

    cur_pos = (6.40, 2.35, -2.80)
    pybullet.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=90, cameraPitch=-30, cameraTargetPosition=cur_pos)
    
    plane_id = pybullet.loadURDF("plane.urdf", basePosition=(5.0, 5.0, -2.80), globalScaling=2)
    drone = pybullet.loadURDF("duck_vhacd.urdf", basePosition=cur_pos, baseOrientation=pybullet.getQuaternionFromEuler([1.57079632679, 0, 3.1415926536]), useFixedBase=False, globalScaling=5)
    
    target_pos = (-0.20, 2.15, -1.85)
    target = pybullet.loadURDF("soccerball.urdf", basePosition=target_pos, useFixedBase=True, globalScaling=0.5)
    
    obstacles = [((random.uniform(*x_bounds), random.uniform(*y_bounds), random.uniform(*z_bounds)), 0.75) for _ in range(10)]
    for ob in obstacles:
        pybullet.loadURDF("sphere2.urdf", basePosition=ob[0], useFixedBase=True, globalScaling=ob[1])

    attract_coeff = 50
    repul_coeff = 20
    global_delta = vector_magnitude(vector_subtract(cur_pos, target_pos))

    while True:
        try:
            if not pybullet.isConnected(client):
                print("Disconnected from PyBullet physics server.")
                break
            
            cur_pos, cur_orient = pybullet.getBasePositionAndOrientation(drone, physicsClientId=client)
            local_delta = vector_magnitude(vector_subtract(cur_pos, target_pos))
    
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
            velocity_x = total_force[0] / local_delta * scalar
            velocity_y = total_force[1] / local_delta * scalar
            velocity_z = total_force[2] / local_delta * scalar

            print("Control Velocity :", (velocity_x, velocity_y, velocity_z))
            print("Current Position :", cur_pos)
            print("Target Position :", target_pos)
            
            print()
            
            pybullet.resetBaseVelocity(
                drone, (velocity_x, velocity_y, velocity_z), (0, 0, 0)
            )

            dist = vector_magnitude(vector_subtract(cur_pos, target_pos))
            if dist <= 0.3:
                print("Success: Drone reached the target!")

            pybullet.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=90, cameraPitch=-30, cameraTargetPosition=cur_pos)
            
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
        pos = [list(map(float, line.split()[2:])) for line in data]
    
    client = pybullet.connect(pybullet.GUI)
    if client < 0:
        raise RuntimeError("Failed to connect to PyBullet physics server.")
    
    pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())

    cur_pos = config["takeoff_pos"]
    pybullet.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=90, cameraPitch=-30, cameraTargetPosition=cur_pos)
    
    plane_id = pybullet.loadURDF("plane.urdf", basePosition=config["takeoff_pos"], globalScaling=2)
    drone = pybullet.loadURDF("duck_vhacd.urdf", basePosition=cur_pos, baseOrientation=pybullet.getQuaternionFromEuler([1.57079632679, 0, 3.1415926536]), useFixedBase=False, globalScaling=5)
    
    target_pos = config["target_pos"]
    target = pybullet.loadURDF("soccerball.urdf", basePosition=target_pos, useFixedBase=True, globalScaling=0.5)
    
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
                drone, cur_pos[1:], cur_orient
            )

            dist = vector_magnitude(vector_subtract(cur_pos, target_pos))
            if dist <= 0.3:
                print("Success: Drone reached the target!")
                return log_dir

            pybullet.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=90, cameraPitch=-30, cameraTargetPosition=cur_pos[1:])
            pybullet.stepSimulation()
            time.sleep(0.025)
        except Exception as e:
            print(traceback.format_exc())
            break
    pybullet.disconnect()

if __name__ == "__main__":
    # success_lst = []
    # logs = sorted(os.listdir("logs"))
    
    # for i in range(len(logs)):
    #     res = test_log("logs/"+logs[i]+"/")
    #     if res:
    #         success_lst.append(res)
            
    # print(success_lst)
    
    test_apf()