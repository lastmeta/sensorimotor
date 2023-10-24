''' simple infinite environment '''

import numpy as np
import gym

from sensorimotor.envs.number_line import NumberLine


class FiniteNumberLine(NumberLine):
    ''' the agent can move back and forth on the number line '''
    metadata = {'render.modes': ['human']}

    def __init__(self, max=100, min=0, gapMin=45, gapMax=51):
        super().__init__()
        self.max = max
        self.min = min
        self.gapMin = gapMin
        self.gapMax = gapMax

    def _calculate_state(self, action):
        self.prior = self.state
        proposition = self.state + action
        # boundary
        if proposition > self.max:
            self.state = self.max
        elif proposition < self.min:
            self.state = self.min
        # gap
        elif (
            self.gapMin < self.gapMax and
            self.gapMin < proposition < self.gapMax and
            self.state <= self.gapMin
        ):
            self.state = self.gapMin
        elif (
            self.gapMin < self.gapMax and
            self.gapMin < proposition < self.gapMax and
            self.state >= self.gapMax
        ):
            self.state = self.gapMax
        else:
            self.state += action

    def execute(self, actions):
        for action in actions:
            if isinstance(action, int):
                action = {0: 0, 1: 1, 2: -1, 3: 10, 4: -9}.get(action, 0)
            else:
                action = 0
            self._calculate_state(action)
        return self.state


'''
Theory: This is what we hope to see:

    on the infinite numberline the agent automatically deduces a set of rules
    that are generalized to understand and manipulate the space.

    on the finite numberline the agent automatically deduces those rules above,
    along with exceptions to those rules that define certain instances like the
    gap and the boundaries.
'''
