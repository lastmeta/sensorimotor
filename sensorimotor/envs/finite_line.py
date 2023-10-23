''' simple infinite environment '''

import numpy as np
import gym

from sensorimotor.envs.number_line import NumberLine


class FiniteNumberLine(NumberLine):
    ''' the agent can move back and forth on the number line '''
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()

    def _calculate_state(self, action):
        proposition = self.state + action
        # boundary
        if proposition > 100:
            self.state = 100
        elif proposition < 0:
            self.state = 0
        # gap
        elif 45 < proposition < 51 and self.state <= 45:
            self.state = 45
        elif 45 < proposition < 51 and self.state >= 51:
            self.state = 51
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
