# unused - we use Gym instead.
'''
agents communicate in binary strings (motor output, actions)
environments communicate in binary strings (state rep, observaiton)
encoded binary strings are translated to integers (symbols) then mapped to 
integers (actual affects)
action pipeline example: 00000010 -> 2 -> -1
state pipeline example: 00000011 -> 3, 
state change pipeline: 3 + -1 = 2, 2 -> 00000010

Of course, if the environment can manipulate binary strings directly, using a nn
or something, then the user of this class can simply override the decode and
encode functions to be the identity function.
'''


class Environment():
    ''' the agent can move back and forth on the number line '''

    def __init__(
        self,
        initial_state: str = None, initial_action: str = None,
        initial_decoded_state: str = None, initial_decoded_action: str = None
    ):
        ''' initial_state: binary string, initial_action: binary string,
            initial_decoded_state: int, initial_decoded_action: int '''
        super(Environment, self).__init__()
        self.initial_state = initial_decoded_state or self._decode_state(
            initial_state)
        self.initial_action = initial_decoded_action or self._decode_action(
            initial_action)
        self.reset()

    def reset(self):
        self.state = self.initial_state
        self.action = self.initial_action

    def execute(self, actions: str):
        for action in actions:
            self.step(action)
        return self.state

    def step(self, action):
        return self._request(self._decode_action(action))

    def render(self):
        print("{}\t{}\t".format(self.action, self.state))

    # default encode translates binary to integers, any specific environment
    # would have a more sophisticated, a more semantic tranlsation
    def _decode_action(self, bin: str):
        ''' accepts a binary string of a certain size and returns int action '''
        return int(bin, 2)

    # default encode translates binary to integers, any specific environment
    # would have a more sophisticated, a more semantic tranlsation
    def _encode_action(self, integer: int, bin_size: int = 8):
        ''' converts an integer to a binary string of bin_size '''
        return bin(integer)[2:].zfill(bin_size)

    # default encode translates binary to integers, any specific environment
    # would have a more sophisticated, a more semantic tranlsation
    def _decode_state(self, bin: str):
        ''' accepts a binary string of a certain size and returns int action '''
        return int(bin, 2)

    # default encode translates binary to integers, any specific environment
    # would have a more sophisticated, a more semantic tranlsation
    def _encode_state(self, integer: int, bin_size: int = 8):
        ''' converts an integer to a binary string of bin_size '''
        return bin(integer)[2:].zfill(bin_size)

    # once the action is translated to an symbol (like an integer)
    # it can be simply mapped to a translation of the state
    def _action_affect(self, action: int):
        ''' returns what we're going to do to the state '''
        if isinstance(action, int):
            return {0: 0, 1: 1, 2: -1}.get(action, 0)
        else:
            return 0

    # here we apply the action to the state
    def _state_behavior(self, affect: int):
        affect = (affect or 0)
        if self.state is None:
            return 0 + affect
        return self._decode_state(self.state) + affect

    def _request(self, action: int):
        self.state = self._encode_state(
            self._state_behavior(
                self._action_affect(action)))
        return self.state
