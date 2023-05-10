from pettingzoo.test import api_test
from crowd_rl import crowd_v1
env = crowd_v1.env()
api_test(env, num_cycles=1000, verbose_progress=False)