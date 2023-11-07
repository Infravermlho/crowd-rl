import copy
import random

import dill
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def custom_getAction(
    self,
    states,
    epsilon=0,
    agent_mask=None,
    env_defined_actions=None,
    action_masks=None,
):
    """Returns the next action to take in the environment.
    Epsilon is the probability of taking a random action, used for exploration.
    For epsilon-greedy behaviour, set epsilon to 0.

    :param state: Environment observations: {'agent_0': state_dim_0, ..., 'agent_n': state_dim_n}
    :type state: Dict[str, numpy.Array]
    :param epsilon: Probablilty of taking a random action for exploration, defaults to 0
    :type epsilon: float, optional
    :param agent_mask: Mask of agents to return actions for: {'agent_0': True, ..., 'agent_n': False}
    :type agent_mask: Dict[str, bool]
    :param env_defined_actions: Mask of agents to return actions for: {'agent_0': True, ..., 'agent_n': False}
    :type env_defined_actions: Dict[str, bool]
    """
    # Get agents, states and actions we want to take actions for at this timestep according to agent_mask
    if agent_mask is None:
        agent_ids = self.agent_ids
        actors = self.actors
    else:
        agent_ids = [agent for agent in agent_mask.keys() if agent_mask[agent]]
        states = {
            agent: states[agent] for agent in agent_mask.keys() if agent_mask[agent]
        }
        actors = [
            actor
            for agent, actor in zip(agent_mask.keys(), self.actors)
            if agent_mask[agent]
        ]

    # Convert states to a list of torch tensors
    states = [torch.from_numpy(state).float() for state in states.values()]
    action_masks = [action_mask for action_mask in action_masks.values()]

    # Configure accelerator
    if self.accelerator is None:
        states = [state.to(self.device) for state in states]

    if self.one_hot:
        states = [
            nn.functional.one_hot(state.long(), num_classes=state_dim[0])
            .float()
            .squeeze()
            for state, state_dim in zip(states, self.state_dims)
        ]

    if self.arch == "mlp":
        states = [
            state.unsqueeze(0) if len(state.size()) < 2 else state for state in states
        ]
    elif self.arch == "cnn":
        states = [state.unsqueeze(2) for state in states]

    actions = {}
    for idx, (agent_id, state, action_mask, actor) in enumerate(
        zip(agent_ids, states, action_masks, actors)
    ):
        if random.random() < epsilon:
            if self.discrete_actions:
                if action_mask is None:
                    action = np.random.randint(0, self.action_dims[idx])
                else:
                    inv_mask = 1 - action_mask

                    available_actions = np.ma.array(
                        np.arange(0, self.action_dims[idx]), mask=inv_mask
                    ).compressed()
                    action = np.random.choice(available_actions)
                    # print(f"action_mask: {action_mask}")
                    # print(f"available_actions: {available_actions}")
                    # print(f"action: {action}")
            else:
                action = (
                    np.random.rand(state.size()[0], self.action_dims[idx])
                    .astype("float32")
                    .squeeze()
                )
        else:
            actor.eval()
            if self.accelerator is not None:
                with actor.no_sync():
                    action_values = actor(state)
            else:
                with torch.no_grad():
                    action_values = actor(state)
            actor.train()

            if self.discrete_actions:
                if action_mask is None:
                    action = action_values.squeeze(0).argmax().item()
                else:
                    exit()
                    inv_mask = 1 - action_mask
                    masked_action_values = np.ma.array(
                        action_values.cpu().data.numpy(), mask=inv_mask
                    )
                    action = np.argmax(masked_action_values, axis=-1)
            else:
                action = action_values.cpu().data.numpy().squeeze() + np.random.normal(
                    0,
                    self.max_action[idx][0] * self.expl_noise,
                    size=self.action_dims[idx],
                ).astype(np.float32)
                action = np.clip(
                    action, self.min_action[idx][0], self.max_action[idx][0]
                )
        # print(f"final_action: {action}")
        actions[agent_id] = action

    if env_defined_actions is not None:
        for agent in env_defined_actions.keys():
            if not agent_mask[agent]:
                actions.update({agent: env_defined_actions[agent]})

    return actions
