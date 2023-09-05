from crowd_rl import crowd_rl_v0 as crowd

if __name__ == "__main__":
    env = crowd.env(render_mode="human")
    env.reset(seed=42)

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            mask = observation["action_mask"]
            # this is where you would insert your policy
            action = env.action_space(agent).sample(mask)

        env.step(action)
        
env.close()