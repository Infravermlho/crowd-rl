"""
MATD3 agent 
"""
import os

import numpy as np
import torch
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.utils import initialPopulation
from tqdm import trange

# from pettingzoo.mpe import simple_speaker_listener_v4
from crowd_rl import crowd_rl_v0

from train_config import env_config
from agile_custom import custom_getAction, custom_test


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("===== AgileRL Online Multi-Agent Demo =====")

    # Define the network configuration
    NET_CONFIG = {
        "arch": "mlp",  # Network architecture
        "h_size": [32, 32],  # Actor hidden size
    }

    # Define the initial hyperparameters
    INIT_HP = {
        "POPULATION_SIZE": 6,
        "ALGO": "MATD3",  # Algorithm
        # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
        "CHANNELS_LAST": False,
        "BATCH_SIZE": 1024,  # Batch size
        "LR": 0.01,  # Learning rate
        "GAMMA": 0.95,  # Discount factor
        "MEMORY_SIZE": 100000,  # Max memory buffer size
        "LEARN_STEP": 5,  # Learning frequency
        "TAU": 0.01,  # For soft update of target parameters
        "POLICY_FREQ": 2,  # Policy frequnecy
    }

    # Define the simple speaker listener environment as a parallel environment
    env = crowd_rl_v0.parallel_env(config=env_config)
    env.reset()

    # Configure the multi-agent algo input arguments
    state_dim = [
        env.observation_space(agent)["observation"].shape for agent in env.agents
    ]

    one_hot = False
    action_dim = [env.action_space(agent).n for agent in env.agents]
    INIT_HP["DISCRETE_ACTIONS"] = True
    INIT_HP["MAX_ACTION"] = None
    INIT_HP["MIN_ACTION"] = None

    # Append number of agents and agent IDs to the initial hyperparameter dictionary
    INIT_HP["N_AGENTS"] = env.num_agents
    INIT_HP["AGENT_IDS"] = env.agents

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

    # Configure the multi-agent replay buffer
    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = MultiAgentReplayBuffer(
        INIT_HP["MEMORY_SIZE"],
        field_names=field_names,
        agent_ids=INIT_HP["AGENT_IDS"],
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
        architecture=0.2,  # Probability of architecture mutation
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
        agent_ids=INIT_HP["AGENT_IDS"],
        arch=NET_CONFIG["arch"],
        rand_seed=1,
        device=device,
    )

    # Define training loop parameters
    max_episodes = 2000  # Total episodes (default: 6000)
    max_steps = 50  # Maximum steps to take in each episode
    epsilon = 0.9  # Starting epsilon value
    eps_end = 0.1  # Final epsilon value
    eps_decay = 0.995  # Epsilon decay
    evo_epochs = 20  # Evolution frequency
    evo_loop = 1  # Number of evaluation episodes
    elite = pop[0]  # Assign a placeholder "elite" agent

    # Training loop
    for idx_epi in trange(max_episodes):
        for agent in pop:
            state, info = env.reset()

            action_mask = {i: state[i]["action_mask"] for i in state}
            observation = {i: state[i]["observation"] for i in state}

            agent_reward = {agent_id: 0 for agent_id in env.agents}
            for _ in range(max_steps):
                agent_mask = info["agent_mask"]
                env_defined_actions = info["env_defined_actions"]

                action = custom_getAction(
                    agent,
                    observation,
                    epsilon,
                    action_masks=action_mask,
                    agent_mask=agent_mask,
                    env_defined_actions=env_defined_actions,
                )  # Get next action from agent

                next_state, reward, termination, truncation, info = env.step(
                    action
                )  # Act in environment

                next_observations = {
                    i: next_state[i]["observation"] for i in next_state
                }
                next_action_mask = {i: next_state[i]["action_mask"] for i in next_state}


                # Save experiences to replay buffer
                memory.save2memory(
                    observation, action, reward, next_observations, termination
                )

                # Collect the reward
                for agent_id, r in reward.items():
                    agent_reward[agent_id] += r

                # Learn according to learning frequency
                if (memory.counter % agent.learn_step == 0) and (
                    len(memory) >= agent.batch_size
                ):
                    experiences = memory.sample(
                        agent.batch_size
                    )  # Sample replay buffer
                    agent.learn(experiences)  # Learn according to agent's RL algorithm

                # Stop episode if any agents have terminated
                if any(truncation.values()) or any(termination.values()):
                    print("reached the end")
                    break

                state = next_state
                action_mask = next_action_mask
                observation = next_observations

            # Save the total episode reward
            score = sum(agent_reward.values())
            agent.scores.append(score)

        # Update epsilon for exploration
        # epsilon = max(eps_end, epsilon - eps_decay)
        epsilon = max(eps_end, epsilon * eps_decay)
        print(f"new epsilon value {epsilon}")

        # Now evolve population if necessary
        if (idx_epi + 1) % evo_epochs == 0:
            # Evaluate population
            fitnesses = [
                custom_test(
                    agent,
                    env,
                    swap_channels=INIT_HP["CHANNELS_LAST"],
                    loop=evo_loop,
                )
                for agent in pop
            ]

            print(f"Episode {idx_epi + 1}/{max_episodes}")
            print(f'Fitnesses: {["%.2f" % fitness for fitness in fitnesses]}')
            print(
                f'100 fitness avgs: {["%.2f" % np.mean(agent.fitness[-100:]) for agent in pop]}'
            )

            # Tournament selection and population mutation
            elite, pop = tournament.select(pop)
            pop = mutations.mutation(pop)

    # Save the trained algorithm
    path = "./models/MATD3"
    filename = "MATD3_trained_agent.pt"
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, filename)
    elite.saveCheckpoint(save_path)
