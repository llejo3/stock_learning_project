import random
import gym
import reinforcement.env

from reinforcement.agent.agent_params import AgentParams


class RandomAgent:

    def __init__(self, action_space):
        self.params = AgentParams()
        self.action_space = action_space

    def act(self):
        return self.action_space.sample()


def main():
    env = gym.make('StockEnv-v0')
    env.set_corp(corp_name='카카오')
    env.reset()
    agent = RandomAgent(env.action_space)
    while True:
        action = agent.act()
        ob, reward, done, info = env.step(action)
        print(info)
        if done:
            break

if __name__ == "__main__":
    main()
