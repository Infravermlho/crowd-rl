from pettingzoo.test import performance_benchmark, parallel_api_test
from pettingzoo.butterfly import pistonball_v6
# env = pistonball_v6.env()
# performance_benchmark(env)


# ----------------------------------------------------------------------

from crowd_rl import crowd_rl_v0 as crowd
from test_config import env_config
import time

env = crowd.parallel_env(config=env_config)
parallel_api_test(env, num_cycles=1000)


# from pettingzoo.mpe import simple_speaker_listener_v4
# env = simple_speaker_listener_v4.env(continuous_actions=True)
# performance_benchmark(env)