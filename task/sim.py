from functools import partial
from tellodrone_sim import TelloDroneSim

log = False

t = TelloDroneSim()

if log:
    t.active_task = t.run_log
    t.load_config("logs/log-25-11-2024_10:42:13/")
else:
    t.attract_coeff = 20
    t.repel_coeff = 5
    t.influence_dist = 0.5
    t.bounds_influence_dist = 0.75

    t.active_task = partial(t.run_apf, bounds=True)
    t.generate_obstacles(10, 0.75)

t.start_sim()
t.run_sim()