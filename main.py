from Agent import Agent
from Monitor import interact
import gym

env = gym.make('Taxi-v2')
agent = Agent()
avg_rewards, best_avg_reward = interact(env, agent)