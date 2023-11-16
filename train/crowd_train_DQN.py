from train_config import env_config
from agilerl.components.replay_buffer import ReplayBuffer
import numpy as np
from agilerl.algorithms.dqn_rainbow import RainbowDQN

from crowd_rl import crowd_rl_v0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the network configuration
NET_CONFIG = {
    "arch": "cnn",  # Network architecture
    "h_size": [64, 64],  # Actor hidden size
    "c_size": [128],  # CNN channel size
    "k_size": [4],  # CNN kernel size
    "s_size": [1],  # CNN stride size
    "normalize": False,  # Normalize image from range [0,255] to [0,1]
}

# Define the initial hyperparameters
INIT_HP = {
    "POPULATION_SIZE": 6,
    # "ALGO": "Rainbow DQN",  # Algorithm
    "ALGO": "DQN",  # Algorithm
    "DOUBLE": True,
    # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
    "BATCH_SIZE": 256,  # Batch size
    "LR": 1e-4,  # Learning rate
    "GAMMA": 0.99,  # Discount factor
    "MEMORY_SIZE": 100000,  # Max memory buffer size
    "LEARN_STEP": 1,  # Learning frequency
    "N_STEP": 1,  # Step number to calculate td error
    "PER": False,  # Use prioritized experience replay buffer
    "ALPHA": 0.6,  # Prioritized replay buffer parameter
    "TAU": 0.01,  # For soft update of target parameters
    "BETA": 0.4,  # Importance sampling coefficient
    "PRIOR_EPS": 0.000001,  # Minimum priority for sampling
    "NUM_ATOMS": 51,  # Unit number of support
    "V_MIN": 0.0,  # Minimum value of support
    "V_MAX": 200.0,  # Maximum value of support
}


# Create environment and Experience Replay Buffer
env = crowd_rl_v0.env(config=env_config)
env.reset()

state_dim = [env.observation_space(agent)["observation"].shape for agent in env.agents]
one_hot = False
action_dim = [env.action_space(agent).n for agent in env.agents]
INIT_HP["DISCRETE_ACTIONS"] = True
INIT_HP["MAX_ACTION"] = None
INIT_HP["MIN_ACTION"] = None

state_dim = np.moveaxis(np.zeros(state_dim[0]), [-1], [-3]).shape
action_dim = action_dim[0]

# Create a population ready for evolutionary hyper-parameter optimisation
pop = initialPopulation(
    INIT_HP["ALGO"],
    state_dim,
    action_dim,
    one_hot,
    NET_CONFIG,
    INIT_HP,
    population_size=INIT_HP["POPULATION_SIZE"],
    device=device,
)

# Configure the replay buffer
field_names = ["state", "action", "reward", "next_state", "done"]
memory = ReplayBuffer(
    action_dim=action_dim,  # Number of agent actions
    memory_size=INIT_HP["MEMORY_SIZE"],  # Max replay buffer size
    field_names=field_names,  # Field names to store in memory
    device=device,
)

# Instantiate a tournament selection object (used for HPO)
tournament = TournamentSelection(
    tournament_size=2,  # Tournament selection size
    elitism=True,  # Elitism in tournament selection
    population_size=INIT_HP["POPULATION_SIZE"],  # Population size
    evo_step=1,
)  # Evaluate using last N fitness scores


# Instantiate a mutations object (used for HPO)
mutations = Mutations(
    algo=INIT_HP["ALGO"],
    no_mutation=0.2,  # Probability of no mutation
    architecture=0,  # Probability of architecture mutation
    new_layer_prob=0.2,  # Probability of new layer mutation
    parameters=0.2,  # Probability of parameter mutation
    activation=0,  # Probability of activation function mutation
    rl_hp=0.2,  # Probability of RL hyperparameter mutation
    rl_hp_selection=[
        "lr",
        "learn_step",
        "batch_size",
    ],  # RL hyperparams selected for mutation
    mutation_sd=0.1,  # Mutation strength
    # Define search space for each hyperparameter
    min_lr=0.0001,
    max_lr=0.01,
    min_learn_step=1,
    max_learn_step=120,
    min_batch_size=8,
    max_batch_size=64,
    arch=NET_CONFIG["arch"],  # MLP or CNN
    rand_seed=1,
    device=device,
)

# Define training loop parameters
episodes_per_epoch = 10
max_episodes = 10  # Total episodes
max_steps = 500  # Maximum steps to take in each episode
evo_epochs = 20  # Evolution frequency
evo_loop = 50  # Number of evaluation episodes
elite = pop[0]  # Assign a placeholder "elite" agent
epsilon = 1.0  # Starting epsilon value
eps_end = 0.1  # Final epsilon value
eps_decay = 0.9998  # Epsilon decays
opp_update_counter = 0


while True:
    action = agent.getAction(state, epsilon)  # Get next action from agent
    next_state, reward, done, _, _ = env.step(action)  # Act in environment

    # Save experience to replay buffer
    if channels_last:
        memory.save2memoryVectEnvs(
            state, action, reward, np.moveaxis(next_state, [3], [1]), done
        )
    else:
        memory.save2memoryVectEnvs(state, action, reward, next_state, done)

    # Learn according to learning frequency
    if memory.counter % agent.learn_step == 0 and len(memory) >= agent.batch_size:
        experiences = memory.sample(agent.batch_size)  # Sample replay buffer
        agent.learn(experiences)  # Learn according to agent's RL algorithm
