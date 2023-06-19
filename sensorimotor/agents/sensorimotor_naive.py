'''
implementation of the naive agent as a benchmark for toy exmaples
'''

import time
import anytree

class NaiveSensorimotor(object):
    ''' suitable for small, simple environments. uses a tree - explicit memory '''

    def __init__(self, env):
        self.env = env
        self.root = anytree.Node('root')
        self.previous = self.root
        self.action = 0


    def memorize(self, obs):
        ''' every time I see an observation add it to the tree pointing to the previous one '''
        if self.previous and not isinstance(self.previous, int):
            node = anytree.Node(obs, parent=self.previous, edge=self.action)
        self.previous = node

    def random_step(self, obs):
        self.memorize(obs)
        self.action = self.env.action_space.sample()
        return self.action

    def get_path(self, target, start=None):
        ''' recursive breadth first search from both ends '''

        def remove_do_nothings(actions, do_nothing=None):
            return [a for a in actions if a != do_nothing]

        start = start or (self.previous if isinstance(self.previous, int) else self.previous.name)
        targets = anytree.search.findall(self.root, filter_=lambda node: node.name == target)
        shortest = None
        shortest_length = 1_000_000
        for t in targets:
            i = 0
            found = False
            for node in t.iter_path_reverse():
                i += 1
                if node.name == start:
                    found = True
                    break
            if found and i < shortest_length:
                shortest = t
                shortest_length = i
        if shortest_length == 0 and found == False:
            return 'not found'
        actions = []
        for node in shortest.iter_path_reverse():
            actions.append(node.edge)
            if node.name == start:
                break
        return remove_do_nothings([i for i in reversed(actions)][1:], do_nothing=0)

    def reset(self, state):
        self.env.reset(state=state)
        self.previous = state
        return self.previous

    def do(self, state, verbose=False):
        actions = self.get_path(target=state)
        if verbose:
            print(actions)
        return self.env.execute(actions=actions)

    def train(self, epocs=1, steps=1000, verbose=False):
        for _ in range(epocs):
            obs = self.env.reset()
            if verbose:
                self.env.render()
            for _ in range(steps):
                action = self.random_step(obs)
                obs, _reward, _done, _info = self.env.step(action)
                if verbose:
                    # notice its moving through the environment state-space...
                    time.sleep(.001)
                    print(obs, end='\r')