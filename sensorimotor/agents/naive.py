'''
implementation of the naive agent as a benchmark for toy exmaples
this implementation uses the graph object which records the graph more 
efficiently than the tree implementation but is slower at path finding.
the path finding is optimized by breath-frist-search from both ends.
'''
from typing import Union
from sensorimotor import Graph


class NaiveAgent(object):
    ''' suitable for small, simple environments. uses a tree - explicit memory '''

    def __init__(self, env, state=None):
        self.env = env
        self.graph = Graph()
        self.state = None
        if state is not None:
            self.seed(state)

    def seed(self, state):
        ''' the first state '''
        if self.state == None:
            self.memorize(state, action=0)
        else:
            self.reset('root')
            self.memorize(state, action=0)
        self.state = state

    def memorize(self, child: str, action: Union[str, int], parent: str = None):
        ''' save to graph '''
        self.graph.add(
            parent=parent or self.state,
            child=child,
            edge=action)

    def random_step(self):
        return self.env.action_space.sample(), None

    def new_random_step(self):
        new = False
        sisters = self.graph.get_children(parent=self.state)
        sisterActions = [v for v in sisters.values()]
        if len(sisterActions) >= self.env.action_space.n:
            action = self.env.action_space.sample()
        else:
            new = True
            while True:
                action = self.env.action_space.sample()
                if action not in sisterActions:
                    break
        return action, new

    def get_path(self, target, start=None, simply=False):
        if simply:
            fn = self.graph.get_path_only_from_parent
        else:
            fn = self.graph.path
        path = fn(parent=start or self.state, child=target)
        if path is not None:
            return [fromToAction[-1] for fromToAction in path]
        return path

    def reset(self, state):
        self.env.reset(state=state)
        self.state = state
        return self.state

    def do(self, state, verbose=False):
        actions = self.get_path(target=state)
        if verbose:
            print(actions)
        return self.env.execute(actions=actions)

    def fully_train(self, epocs=1, steps=1000, verbose=False, extraVerbose=False):
        # though we wrote this learnedSomethingNew metric to be detected at
        # action selection, it's not behaving as expected. So we'll use a simpler metric to
        # determin if we can stop: the number of pairs in the graph doesn't change.
        learnedSomethingNew = True
        i = 0
        priorPairCount = 0
        while learnedSomethingNew:
            if extraVerbose:
                print('Iteration:', i, 'Graph Size:', len(self.graph.pairs))
            if verbose:
                print('.', end='')
            learnedSomethingNew = self.train(
                epocs=epocs,
                steps=steps,
                verbose=extraVerbose,
                extraVerbose=False)
            if len(self.graph.pairs) == priorPairCount:
                break
            priorPairCount = len(self.graph.pairs)
            i += 1
        print('Iteration:', i, 'Graph Size:', len(self.graph.pairs))

    def train(self, epocs=1, steps=1000, verbose=False, extraVerbose=False):
        learnedSomethingNew = 0
        if self.state == None:
            self.state = self.env.reset()
        for _ in range(epocs):
            # state = self.env.reset() # why reset?
            if extraVerbose:
                self.env.render()
            elif verbose:
                print(self.state, end='\r')
            for _ in range(steps):
                action, new = self.new_random_step()
                state, _reward, _done, _info = self.env.step(action)
                self.memorize(parent=self.state, child=state, action=action)
                self.state = state
                learnedSomethingNew = (
                    learnedSomethingNew + 1) if new else learnedSomethingNew
        return learnedSomethingNew
