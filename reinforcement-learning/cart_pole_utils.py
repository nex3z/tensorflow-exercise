import gym
import numpy as np


def play_one_episode(policy, render=False, seed=47):
    frames = []
    rewards = 0
    env = gym.make("CartPole-v1")
    env.seed(seed)
    np.random.seed(seed)

    obs = env.reset()
    for step in range(200):
        if render:
            frames.append(env.render(mode='rgb_array'))
        action = policy(obs)
        obs, reward, done, info = env.step(action)
        rewards += reward
        if done:
            break
    env.close()
    if render:
        return rewards, frames
    else:
        return rewards
