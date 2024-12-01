from functools import partial
from tellodrone_sim import TelloDroneSim
from vector import Vector3D

log = False

t = TelloDroneSim()

t.attract_coeff = 20
t.repel_coeff = 10
t.influence_dist = 0.5
t.bounds_influence_dist = 0.5

# t.generate_obstacles(5, 0.75)
# t.plan_sine_path(num=20, amplitude=2, frequency=3)
# t.a_star_waypoints(grid_resolution=0.5)

t.start_pos = Vector3D.from_arr([6.168000221252441, 2.187000036239624, -2.1480000019073486])
t.target_pos = Vector3D.from_arr([-0.3, 1.75, -2.0])
t.obstacles = [[[3.6797160387039183, 1.3423776471373428, -1.8293415215265678], 0.40662141760415554], [[4.158914017677307, 1.4281798326701263, -1.75810780208686], 0.36500537657864285], [[3.5350915670394896, 2.7103671975546493, -3.284154669570988], 0.3404207305843513]]
t.obstacles = [(Vector3D.from_arr(pos), rad) for pos, rad in t.obstacles]
t.path = [t.start_pos, t.target_pos]

# t.path = [
#     Vector3D(t.x_bounds[0] - 0.1, 2.0, -2.0), 
#     Vector3D(t.x_bounds[1] + 0.1, 2.0, -2.0), 
#     Vector3D(3.0, t.y_bounds[0] - 0.1, -2.0), 
#     Vector3D(3.0, t.y_bounds[1] + 0.1, -2.0), 
#     Vector3D(3.0, 2.0, t.z_bounds[0] - 0.1), 
#     Vector3D(3.0, 2.0, t.z_bounds[1] + 0.1), 
# ]

t.active_task = t.run_apf
t.start_sim()
t.run_sim()