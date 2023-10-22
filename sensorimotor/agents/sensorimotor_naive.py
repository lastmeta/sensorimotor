'''
implementation of the naive agent as a benchmark for toy exmaples
'''

import time
import anytree


class NaiveSensorimotor(object):
    ''' suitable for small, simple environments. uses a tree - explicit memory '''

    def __init__(self, env, state=None):
        self.env = env
        self.root = anytree.Node('root')
        self.previous = self.root
        self.action = 0
        if state is not None:
            self.seed(state)

    def seed(self, state):
        ''' the first state '''
        if self.previous == self.root:
            self.memorize(state)
        else:
            self.reset('root')
            self.memorize(state)

    def node_exists(self):
        if not self.previous or isinstance(self.previous, int):
            return None
        existing_node = anytree.find(
            self.root,
            lambda node: node.edge == self.action and node.parent == self.previous)
        return existing_node

    def memorize(self, obs):
        ''' every time I see an observation add it to the tree pointing to the previous one '''
        if self.previous and not isinstance(self.previous, int):
            node = anytree.Node(obs, parent=self.previous, edge=self.action)
        self.previous = node

    def random_step(self, obs):
        self.memorize(obs)
        self.action = self.env.action_space.sample()
        return self.action

    def new_random_step(self, obs):
        new = False
        sisters = anytree.search.findall(
            self.root, filter_=lambda node: node.parent == self.previous)
        print(sisters, len(sisters))
        sisterActions = [s.edge for s in sisters]
        if len(sisterActions) >= self.env.action_space.n:
            self.action = self.env.action_space.sample()
        else:
            self.memorize(obs)
            new = True
            while True:
                self.action = self.env.action_space.sample()
                if self.action not in sisterActions:
                    break
        return self.action, new

    def get_path(self, target, start=None):
        ''' recursive breadth first search from both ends '''

        def remove_do_nothings(actions, do_nothing=None):
            return [a for a in actions if a != do_nothing]

        start = start or (self.previous if isinstance(
            self.previous, int) else self.previous.name)
        targets = anytree.search.findall(
            self.root, filter_=lambda node: node.name == target)
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
        if shortest_length == 0 and found == False or shortest == None:
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

    def fully_train(self, epocs=1, steps=1000, verbose=False, extraVerbose=False):
        learnedSomethingNew = True
        while learnedSomethingNew:
            learnedSomethingNew = self.train(
                epocs=epocs,
                steps=steps,
                verbose=verbose,
                extraVerbose=extraVerbose)

    def train(self, epocs=1, steps=1000, verbose=False, extraVerbose=False):
        learnedSomethingNew = False
        for _ in range(epocs):
            obs = self.env.reset()
            if extraVerbose:
                self.env.render()
            elif verbose:
                print(obs, end='\r')
            for _ in range(steps):
                action, new = self.new_random_step(obs)
                learnedSomethingNew = new or learnedSomethingNew
                obs, _reward, _done, _info = self.env.step(action)
        return learnedSomethingNew
