'''
implementation of the naive agent as a benchmark for toy exmaples
this implementation uses the graph object which records the graph more 
efficiently than the tree implementation but is slower at path finding.
the path finding is optimized by breath-frist-search from both ends.
'''
from typing import Union
from sensorimotor import Graph
from random import sample


class NaiveAgent(object):
    ''' suitable for small, simple environments. uses explicit memory '''

    def __init__(self, env, state=None):
        self.env: 'Environment' = env
        self.graph: Graph = Graph()
        if state is not None:
            self.seed(state)

    @property
    def prior(self):
        return self.env.prior

    @property
    def state(self):
        return self.env.state

    @property
    def action(self):
        return self.env.action

    def seed(self, state):
        ''' the first state '''
        self.memorize(child=state, parent=state, action=0)

    def memorize(self, action: Union[str, int] = None, parent: str = None, child: str = None,):
        ''' save to graph '''
        self.graph.add(
            parent=parent or self.prior,
            child=child or self.state,
            edge=action or self.action)

    def random_step(self):
        return self.env.action_space.sample(), None

    def unused_actions(self):
        sisters = self.graph.get_children(parent=self.prior)
        used_nested = [v for v in sisters.values()]
        used_actions = [item for sublist in used_nested for item in sublist]
        if len(used_actions) >= self.env.action_space.n:
            return []
        else:
            return [
                action for action in self.env.actions
                if action not in used_actions]

    def new_random_step(self):
        new = False
        unused = self.unused_actions()
        if len(unused) == 0:
            action = self.env.action_space.sample()
        else:
            new = True
            action = sample(unused, 1)
        return action, new

    def get_path(self, target=None, start=None, simply=False):
        if simply:
            fn = self.graph.get_path_only_from_parent
        else:
            fn = self.graph.path
        path = fn(parent=start or self.prior, child=target or self.state)
        if path is not None:
            return [fromToAction[-1] for fromToAction in path]
        return path

    def reset(self, state):
        self.env.reset(state=state)

    def do(self, state, actions: list = None, verbose=False):
        actions = actions or self.get_path(target=state)
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
                # the state is saved on the environment
                _state, _reward, _done, _info = self.env.step(action)
                self.memorize()
                learnedSomethingNew = (
                    learnedSomethingNew + 1) if new else learnedSomethingNew
        return learnedSomethingNew
