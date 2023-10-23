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

    def render(self, mode='human', close=False):
        action, obs, reward, done, info = self.env_state
        if action == None:
            print("{}\n".format(obs))
        else:
            print("{}\t\t--> {:.18f}{}\n{}\n".format(action,
                  reward, (' DONE!' if done else ''), obs))

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

    def _encode(self, item):
        ''' accepts a binary string of a certain size and returns int action '''
        return int(item, 2)

    def _decode(self, item):
        ''' accepts an integer and returns an binary representation of it.'''
        return bin(item)[2:]

    def _request(self, action):
        if isinstance(action, int):
            action = {0: 0, 1: 1, 2: -1, 3: 10, 4: -9}.get(action, 0)
        else:
            action = 0
        self._calculate_state(action)
        obs = self.state
        # real Intelligence doesn't need spoonfed 'rewards'
        reward = np.float64(0.0)
        done = False
        info = {}
        self.env_state = (action, obs, reward, done, info)
        return obs, reward, done, info

    def _calculate_state(self, action):
        print('WHAT?')
        self.state += action

    def reset(self, state=None):
        if state is None:
            return self._request(None)[0]
        self.state = state
        return state

    def execute(self, actions):
        for action in actions:
            if isinstance(action, int):
                action = {0: 0, 1: 1, 2: -1, 3: 10, 4: -9}.get(action, 0)
            else:
                action = 0
            self.state += action
        return self.state
