''' implementation of the naive agent as a benchmark for toy exmaples '''

import pandas as pd


class NaiveSensorimotor(object):
    ''' suitable for small, simple environments '''

    def __init__(self, env):
        self.env = env
        self.memory = pd.DataFrame(columns=['input', 'action', 'result_id'])
        self.action = 0

    def memorize(self, obs):
        self.memory = self.memory.append({'input':obs, 'action':self.action, 'result_id':self.memory.shape[0]+1}, ignore_index=True)

    def random_step(self, obs):
        self.action = env.action_space.sample()
        self.memorize(obs)
        return self.action


if __name__ == '__main__':
    from sensorimotor.envs import NumberLine
    env = NumberLine()
    env.seed(0)
    print("agent: env.action_space {}".format(env.action_space))
    agent = NaiveSensorimotor(env)
    for i_episode in range(1):
        obs = env.reset()
        env.render()
        for t_timesteps in range(1000):
            action = agent.step(obs)
            obs, reward, done, info = env.step(action)
    env.close()
