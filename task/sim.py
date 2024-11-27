from functools import partial
from tellodrone_sim import TelloDroneSim

log = False

t = TelloDroneSim()

t.attract_coeff = 20
t.repel_coeff = 10
t.influence_dist = 0.5
t.bounds_influence_dist = 0.5
t.generate_obstacles(10, 0.5)
t.plan_sine_path(num=10)
t.active_task = t.run_apf

t.start_sim()
t.run_sim()