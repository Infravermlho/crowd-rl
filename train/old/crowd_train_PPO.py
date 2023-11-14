import gymnasium as gym
from agilerl.utils import makeVectEnvs
from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.algorithms.dqn_rainbow import RainbowDQN

# Create environment and Experience Replay Buffer
env = makeVectEnvs('LunarLander-v2', num_envs=1)
try:
    state_dim = env.single_observation_space.n          # Discrete observation space
    one_hot = True                                      # Requires one-hot encoding
except:
    state_dim = env.single_observation_space.shape      # Continuous observation space
    one_hot = False                                     # Does not require one-hot encoding
try:
    action_dim = env.single_action_space.n              # Discrete action space
except:
    action_dim = env.single_action_space.shape[0]       # Continuous action space

channels_last = False # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]

if channels_last:
    state_dim = (state_dim[2], state_dim[0], state_dim[1])

field_names = ["state", "action", "reward", "next_state", "done"]
memory = ReplayBuffer(action_dim=action_dim, memory_size=10000, field_names=field_names)

agent = RainbowDQN(state_dim=state_dim, action_dim=action_dim, one_hot=one_hot)   # Create agent

state = env.reset()[0]  # Reset environment at start of episode
while True:
    if channels_last:
        state = np.moveaxis(state, [3], [1])
    action = agent.getAction(state, epsilon)    # Get next action from agent
    next_state, reward, done, _, _ = env.step(action)   # Act in environment

    # Save experience to replay buffer
    if channels_last:
        memory.save2memoryVectEnvs(state, action, reward, np.moveaxis(next_state, [3], [1]), done)
    else:
        memory.save2memoryVectEnvs(state, action, reward, next_state, done)

    # Learn according to learning frequency
    if memory.counter % agent.learn_step == 0 and len(memory) >= agent.batch_size:
        experiences = memory.sample(agent.batch_size) # Sample replay buffer
        agent.learn(experiences)    # Learn according to agent's RL algorithm