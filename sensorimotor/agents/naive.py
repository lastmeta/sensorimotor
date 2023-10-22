'''
implementation of the naive agent as a benchmark for toy exmaples
this implementation uses the graph object which records the graph more 
efficiently than the tree implementation but is slower at path finding.
the path finding is optimized by breath-frist-search from both ends.
'''

from sensorimotor import Graph


class NaiveAgent(object):
    ''' suitable for small, simple environments. uses a tree - explicit memory '''

    def __init__(self, env, state=None):
        self.env = env
        self.graph = Graph()
        self.previous = None
        self.action = 0
        if state is not None:
            self.seed(state)

    def seed(self, state):
        ''' the first state '''
        if self.previous == None:
            self.memorize(state)
        else:
            self.reset('root')
            self.memorize(state)

    def memorize(self, state):
        ''' save to graph '''
        self.graph.add(parent=self.previous, child=state, edge=self.action)
        self.previous = state

    def node_exists(self, parent, child, action) -> bool:
        ''' returns true if the node exists in the graph '''

    def random_step(self, state):
        self.memorize(state)
        self.action = self.env.action_space.sample()
        return self.action

    def new_random_step(self, state):
        new = False
        sisters = self.graph.get_children(parent=self.previous)
        sisterActions = [v for v in sisters.values()]
        if len(sisterActions) >= self.env.action_space.n:
            self.action = self.env.action_space.sample()
        else:
            self.memorize(state)
            new = True
            while True:
                self.action = self.env.action_space.sample()
                if self.action not in sisterActions:
                    break
        return self.action, new

    def get_path_simply(self, target, start=None):
        return [fromToAction[-1] for fromToAction in self.graph.get_path_only_from_parent(parent=start or self.previous, child=target)]

    def get_path(self, target, start=None):
        return [fromToAction[-1] for fromToAction in self.graph.path(parent=start or self.previous, child=target)]

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
            state = self.env.reset()
            if extraVerbose:
                self.env.render()
            elif verbose:
                print(state, end='\r')
            for _ in range(steps):
                action, new = self.new_random_step(state)
                learnedSomethingNew = new or learnedSomethingNew
                state, _reward, _done, _info = self.env.step(action)
        return learnedSomethingNew
