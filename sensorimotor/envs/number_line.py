''' simple infinite environment '''

import numpy as np
import gym


class NumberLine(gym.Env):
    ''' the agent can move back and forth on the number line '''
    metadata = {'render.modes': ['human']}
    def __init__(self):
        super(NumberLine, self).__init__()
        self.action_space = self._action_space()
        self.observation_space = self._observation_space()
        self.state = 0

    def step(self, action):
        return self._request(action)

    def reset(self):
        return self._request(None)[0]

    def render(self, mode='human', close=False):
        action, obs, reward, done, info = self.env_state
        if action == None: print("{}\n".format(obs))
        else: print("{}\t\t--> {:.18f}{}\n{}\n".format(action, reward, (' DONE!' if done else ''), obs))

    def _action_space(self):
        '''
        0 = do nothing
        1 = +1
        2 = -1
        3 = +10
        4 = -9
        '''
        return gym.spaces.Discrete(5)

    def _observation_space(self):
        return gym.spaces.Box(low=np.NINF, high=np.inf, shape=(1,), dtype=np.int64)

    def _request(self, action):
        if isinstance(action, int):
            action = {0: 0, 1: 1, 2: -1, 3: 10, 4: -9}.get(action, 0)
        else:
            action = 0
        self.state = self.state + action
        obs = self.state
        reward = np.float64(0.0)  # real AGI doesn't need spoonfed 'rewards'
        done = False
        info = {}
        self.env_state = (action, obs, reward, done, info)
        return obs, reward, done, info
