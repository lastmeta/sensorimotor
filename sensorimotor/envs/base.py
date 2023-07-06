'''
    agents communicate in binary strings (motor output, actions)
    todo: environments communictae in binary strings (state rep, observaiton)
'''


class Environment():
    ''' the agent can move back and forth on the number line '''

    def __init__(self, initial_state: str = None, initial_action: str = None):
        super(Environment, self).__init__()
        self.initial_state = initial_state
        self.initial_action = initial_action
        self.reset()

    def reset(self):
        self.state = self.initial_state
        self.action = self.initial_action

    def execute(self, actions: str):
        for action in actions:
            self.step(action)
        return self.state

    def step(self, action):
        return self._request(self._encode(action))

    def render(self):
        print("{}\t{}\t".format(self.action, self.state))

    # default encode translates binary to integers, any specific environment
    # would have a more sophisticated, a more semantic tranlsation
    def _encode(self, bin: str):
        ''' accepts a binary string of a certain size and returns int action '''
        return int(bin, 2)

    # default encode translates binary to integers, any specific environment
    # would have a more sophisticated, a more semantic tranlsation
    def _decode(self, integer: int, bin_size: int = 8):
        ''' converts an integer to a binary string of bin_size '''
        return bin(integer)[2:].zfill(bin_size)

    # once the action is translated to an symbol (like an integer)
    # it can be simply mapped to a translation of the state
    def _action_behavior(self, action: int):
        if isinstance(action, int):
            return {0: 0, 1: 1, 2: -1}.get(action, 0)
        else:
            return 0

    # here we apply the action to the state
    def _state_behavior(self, action: int):
        return (self.state or 0) + (action or 0)

    def _request(self, action: int):
        self.action = self._action_behavior(action)
        self.state = self._state_behavior(self.action)
        return self.state
