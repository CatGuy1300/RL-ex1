from typing import Tuple

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


def agent(o: np.ndarray[float], w: np.ndarray[float]) -> int:
    product = o@w
    if product >= 0:
        return 1
    else:
        return 0

def initialize_vector() -> np.ndarray[float]:
    w = np.random.uniform(-1, 1, 4)
    return w

def evaluate_agent(w: np.ndarray[float]) -> float:
    accumulated_reward = 0
    env = gym.wrappers.TimeLimit(gym.make("CartPole-v1"), max_episode_steps=200)
    observation, info = env.reset()

    episode_over = False
    while not episode_over:
        action = agent(observation, w)
        observation, reward, terminated, truncated, info = env.step(action)

        accumulated_reward += reward

        episode_over = terminated or truncated

    env.close()
    return accumulated_reward

def random_search() -> Tuple[np.ndarray[float], float, int]:
    best_w = None
    best_reward = 0
    num_of_episodes = 0
    for i in range(10000):
        num_of_episodes += 1
        w = initialize_vector()
        accumulated_reward = evaluate_agent(w)
        if accumulated_reward > best_reward:
            best_reward = accumulated_reward
            best_w = w
        if best_reward >= 200:
            break
    return best_w, best_reward, num_of_episodes

def evaluate_random_search():
    episodes_taken = [random_search()[2] for _ in range(1000)]
    mean = np.mean(episodes_taken)

    plt.hist(episodes_taken, bins=20)
    plt.xlabel('Episodes')
    plt.ylabel('# of episodes')
    plt.title(f'Histogram of Episodes, mean={mean}')
    plt.show()

if __name__ == '__main__':
    evaluate_random_search()
