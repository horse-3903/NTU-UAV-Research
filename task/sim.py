from functools import partial
from tellodrone_sim import TelloDroneSim

log = False

t = TelloDroneSim()

t.attract_coeff = 10
t.repel_coeff = 5
t.influence_dist = 0.5
t.bounds_influence_dist = 0.5
t.generate_obstacles(5, 0.5)
t.plan_sine_path(num=20, amplitude=2, frequency=3)
t.active_task = t.run_apf

t.start_sim()
t.run_sim()