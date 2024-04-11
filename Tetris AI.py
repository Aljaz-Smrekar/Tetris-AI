
import tensorflow as tf
import gymnasium as gym
env = gym.make("ALE/Tetris-v5", render_mode="human")
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()

def __init__(self):
    """Things inside of Tetris"""
    self.game_over = False
    self.reward = 0
    self.line_cleared = 300
    self.tetris = 
def tetris_ai(self):
    """The start of our Tetris AI that will beat the game"""

def scoring(self):
    


def end(self):
    if self.game_over == True:
        return 


