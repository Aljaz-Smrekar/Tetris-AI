
# import gymnasium as gym
# import random
# import math
# import tensorflow
# from tqdm import tqdm
# import matlotlib
from collections import deque
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim

# env = gym.make("ALE/Tetris-v5", render_mode='human')
# env.render()

# #reset teh invoronment of first
# done=False
# obervation, info = env.reset()

# # # Actions: 0 = Noop, 1 = Fire, 2 = Right, 3 = Left, 4 = Down
# action = env.action_space.sample()

# #Execute the action in the environment and recieve the info after takign the step
# observation , reward, terminated, truncated, info = env.step(action)


# class TetrisAi:
#     def __init__(self, learning_rate: float, initial_epsilon: float, final_epsilon: float, discount_factor: float = 0.95):
#         self.lr = learning_rate
#         self.q_value = defaultdict(lambda: np.zeros(env.action_space.n))
#         self.initial_epsilon = initial_epsilon
#         self.final_epsilon = final_epsilon
#         self.discount_factor = discount_factor

#     def train(self, env, n_episodes):
#         for episode in tqdm(range(n_episodes)):
#             obs, info = env.reset()
#             done = False
            
#             while not done:
#                 action = self.get_action(obs)
#                 next_obs, reward, terminated, truncated, info = env.step(action)

#                 self.update(obs, action, reward, terminated, next_obs)
#                 frame = env.render()
#                 plt.imshow(frame)
#                 plt.show()

#                 done = terminated or truncated
#                 obs = self.decay_epsilon(obs)
        


# # # Actions: 0 = Noop, 1 = Fire, 2 = Right, 3 = Left, 4 = Down
# # actions = env.action_space

# # randomAction = env.action_space.sample()
# # reutrnValue = env.step(randomAction)


# # env.render()

# # returnValue = env.step(1)

# # env.reset()


# # env.P[0][1]

# # env.close()


# # # episodes = 1000
# # # for episode in range (1, episodes+1):
# # #     state = env.reset()
# # #     done = False
# # #     score = 0

# # #     while not done:
# # #         action = random.choice([0, 1, 2, 3, 4])
# # #         observation , reward, done, truncated, info = env.step(action)
# # #         score += reward
# # #         env.render()



# # #     print(f"Episode {episode}, Score: {score}")




# # # env.close()







import gym
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.95, initial_epsilon=1.0, final_epsilon=0.1, epsilon_decay=0.995):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay = epsilon_decay

        # Discretize the continuous observation space
        self.obs_bins = self.create_observation_bins(env.observation_space)

        # Initialize the Q-table
        num_states = self.calculate_num_states()
        self.q_table = np.zeros((num_states, env.action_space.n))

    def create_observation_bins(self, obs_space):
        bins = []
        for i in range(len(obs_space.high)):
            bins.append(np.linspace(obs_space.low[i], obs_space.high[i], num=5))  # Adjust the number of bins
        return bins

    def calculate_num_states(self):
        num_states = 1
        for bin_count in self.obs_bins:
            num_states *= (len(bin_count) + 1)
        return int(num_states)

    def discretize_observation(self, obs):
        discretized_obs = []
        for i in range(len(obs)):
            discretized_obs.append(np.digitize(obs[i], bins=self.obs_bins[i]) - 1)
        return tuple(discretized_obs)

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()  # Explore action space randomly
        else:
            return np.argmax(self.q_table[state, :])  # Exploit learned values by choosing optimal action

    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.final_epsilon, self.epsilon)

    def update_q_table(self, state, action, reward, next_state, done):
        if not done:
            next_max = np.max(self.q_table[next_state])
            self.q_table[state, action] += self.learning_rate * (reward + self.discount_factor * next_max - self.q_table[state, action])
        else:
            self.q_table[state, action] += self.learning_rate * (reward - self.q_table[state, action])

    def train(self, num_episodes):
        rewards = []
        for episode in tqdm(range(num_episodes)):
            obs = self.discretize_observation(self.env.reset())
            total_reward = 0
            done = False
            while not done:
                action = self.choose_action(obs)
                next_obs, reward, done, _ = self.env.step(action)
                next_obs = self.discretize_observation(next_obs)
                self.update_q_table(obs, action, reward, next_obs, done)
                total_reward += reward
                obs = next_obs
            rewards.append(total_reward)
            self.decay_epsilon()
        return rewards

# Create the Tetris environment
env = gym.make("ALE/Tetris-v5")

# Create a Q-learning agent
agent = QLearningAgent(env)

# Train the agent
num_episodes = 1000
episode_rewards = agent.train(num_episodes)

# Plot the rewards
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Progress')
plt.show()
