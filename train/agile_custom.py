import random

import numpy as np
import torch
import torch.nn as nn


def custom_getAction(
    self,
    states,
    epsilon=0,
    agent_mask=None,
    env_defined_actions=None,
    action_masks=None,
):
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
                    inv_mask = 1 - action_mask
                    masked_action_values = np.ma.array(
                        action_values.cpu().data.numpy(), mask=inv_mask
                    )
                    action = np.argmax(masked_action_values, axis=-1)[0]
            else:
                action = action_values.cpu().data.numpy().squeeze() + np.random.normal(
                    0,
                    self.max_action[idx][0] * self.expl_noise,
                    size=self.action_dims[idx],
                ).astype(np.float32)
                action = np.clip(
                    action, self.min_action[idx][0], self.max_action[idx][0]
                )
        actions[agent_id] = action

    if env_defined_actions is not None:
        for agent in env_defined_actions.keys():
            if not agent_mask[agent]:
                actions.update({agent: env_defined_actions[agent]})

    return actions


def custom_test(self, env, swap_channels=False, max_steps=500, loop=3):
    with torch.no_grad():
        rewards = []
        for i in range(loop):
            state, info = env.reset()
            action_mask = {i: state[i]["action_mask"] for i in state}
            observation = {i: state[i]["observation"] for i in state}

            agent_reward = {agent_id: 0 for agent_id in self.agent_ids}
            score = 0
            for _ in range(max_steps):
                if swap_channels:
                    state = {
                        agent_id: np.moveaxis(np.expand_dims(s, 0), [3], [1])
                        for agent_id, s in state.items()
                    }
                agent_mask = info["agent_mask"] if "agent_mask" in info.keys() else None
                env_defined_actions = (
                    info["env_defined_actions"]
                    if "env_defined_actions" in info.keys()
                    else None
                )
                action = custom_getAction(
                    self,
                    observation,
                    epsilon=0,
                    agent_mask=agent_mask,
                    action_masks=action_mask,
                    env_defined_actions=env_defined_actions,
                )
                state, reward, done, trunc, info = env.step(action)
                action_mask = {i: state[i]["action_mask"] for i in state}
                observation = {i: state[i]["observation"] for i in state}

                for agent_id, r in reward.items():
                    agent_reward[agent_id] += r
                score = sum(agent_reward.values())

                if any(trunc.values()):
                    break

            rewards.append(score)
    mean_fit = np.mean(rewards)
    self.fitness.append(mean_fit)
    return mean_fit
