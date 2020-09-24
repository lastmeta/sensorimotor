import copy
import time
import numpy as np
np.set_printoptions(precision=8, suppress=True, linewidth=400, threshold=100)
import gym


class RubiksCube(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        super(RubiksCube, self).__init__()
        self.action_space = self._action_space()
        self.observation_space = self._observation_space()
        # should change the state to be a list of np array?
        self.cube_state = {
            1: 'top', 2: 'top', 3: 'top', 4: 'top',
            5: 'top', 6: 'top', 7: 'top', 8: 'top',
            9: 'left',
            10: 'front', 11: 'front', 12: 'front',
            13: 'right', 14: 'right', 15: 'right',
            16: 'back', 17: 'back', 18: 'back',
            19: 'left', 20: 'left', 21: 'left',
            22: 'front', 23: 'front',
            24: 'right', 25: 'right',
            26: 'back', 27: 'back',
            28: 'left', 29: 'left',
            30: 'front', 31: 'front', 32: 'front',
            33: 'right', 34: 'right', 35: 'right',
            36: 'back', 37: 'back', 38: 'back',
            39: 'left', 40: 'left',
            41: 'under', 42: 'under', 43: 'under', 44: 'under',
            45: 'under', 46: 'under', 47: 'under', 48: 'under'}
        self.solved_state = copy.deepcopy(self.cube_state)
        self.do_right = {
            3: 16, 16: 45, 45: 32, 32: 3,
            4: 26, 26: 44, 44: 23, 23: 4,
            5: 36, 36: 43, 43: 12, 12: 5,
            14: 25, 25: 34, 34: 24, 24: 14,
            35: 33, 33: 13, 13: 15, 15: 35, }
        self.do_left = {
            7: 10, 10: 41, 41: 38, 38: 7,
            8: 22, 22: 48, 48: 27, 27: 8,
            1: 30, 30: 47, 47: 18, 18: 1,
            19: 9, 9: 29, 29: 39, 39: 19,
            20: 21, 21: 40, 40: 28, 28: 20, }
        self.do_top = {
            9: 12, 12: 15, 15: 18, 18: 9,
            10: 13, 13: 16, 16: 19, 19: 10,
            11: 14, 14: 17, 17: 20, 20: 11,
            1: 3, 3: 5, 5: 7, 7: 1,
            2: 4, 4: 6, 6: 8, 8: 2, }
        self.do_under = {
            30: 33, 33: 36, 36: 39, 39: 30,
            31: 34, 34: 37, 37: 40, 40: 31,
            32: 35, 35: 38, 38: 29, 29: 32,
            41: 43, 43: 45, 45: 47, 47: 41,
            42: 44, 44: 46, 46: 48, 48: 42, }
        self.do_front = {
            1: 13, 13: 43, 43: 29, 29: 1,
            2: 24, 24: 42, 42: 21, 21: 2,
            3: 33, 33: 41, 41: 9, 9: 3,
            10: 12, 12: 32, 32: 30, 30: 10,
            11: 23, 23: 31, 31: 22, 22: 11, }
        self.do_back = {
            7: 15, 15: 45, 45: 39, 39: 7,
            6: 25, 25: 46, 46: 28, 28: 6,
            5: 35, 35: 47, 47: 19, 19: 5,
            18: 16, 16: 36, 36: 38, 38: 18,
            17: 26, 26: 37, 37: 27, 27: 17, }
    def step(self, action):
        return self._request(action)
    def reset(self):
        return self._request(None)[0]
    def render(self, mode='human', close=False):
        action, obs, reward, done, info = self.state
        if action == None: print("{}\n".format(obs))
        else: print("{}\t\t--> {:.18f}{}\n{}\n".format(action, reward, (' DONE!' if done else ''), obs))

    def _action_space(self):
        '''
        left, right, top, under, front, back
        this is a deterministic env, it doesn't change unless you change it,
        therefore, no opperation isn't available.
        '''
        return gym.spaces.Discrete(6)

    def _observation_space(self):
        return gym.spaces.Box(low=np.NINF, high=np.inf, shape=(48,), dtype=np.float64)

    def _request(self, action):
        cube = copy.deepcopy(self.cube_state)
        action = {
            0: 'left', 1: 'right',
            2: 'top', 3: 'under',
            4: 'front', 5: 'back'}.get(action, None)
        if action is not None:
            for k, v in eval(f'self.do_{action}').items():
                self.cube_state[k] = cube[v]
        obs = self.cube_state
        reward = np.float64(0.0)  # real AGI doesn't need spoonfed 'rewards'
        done = False
        info = {}
        self.state = (action, obs, reward, done, info)
        return obs, reward, done, info
