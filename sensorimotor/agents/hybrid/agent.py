''' https://chat.openai.com/share/7d7c12f6-7880-4083-8a59-16e64ca85f5e '''

import numpy as np
from sensorimotor.agents.hybrid.predict.predictor import Predictor
from sensorimotor.agents.hybrid.predict.pathway import Pathway
from sensorimotor.agents.naive import NaiveAgent


class HybridAgent(NaiveAgent):
    ''' The Hybrid Agent uses predictive models to enhance naive exploration '''

    def __init__(self, env, state=None):
        super().__init__(env, state)
        action_list = list(range(self.env.action_space.n))
        self.predictor = Predictor(
            actions=action_list, dimension=self.state_dimension())
        self.pathway = Pathway(
            actions=action_list, dimension=self.state_dimension())
        self.initial_training_done = False

    def train_predictive_models(self, action, current_state, next_state):
        ''' Train the predictive models with new state transition '''
        self.predictor.train_future_action(
            action,
            np.array([current_state]),
            np.array([next_state]))
        self.predictor.train_past_action(
            action,
            np.array([next_state]),
            np.array([current_state]))

    def train_pathway_models(self):
        ''' Train the predictive models with new state transition '''
        # choose random path from 2 random popular states
        # how do we know which states are popular? we keep a count of every
        # state explored in the graph. to do.
        # target = self.graph.get_random_popular_state()
        # start = self.graph.get_random_popular_state()
        # path = self.get_path(target=target, start=start)
        # middle = path[len(path) // 2]
        # self.pathway.train_middle_state(
        #    current_state=np.array([start]),
        #    goal_state=np.array([target]),
        #    middle_state=middle)

    def state_dimension(self):
        ''' Returns the dimensionality of the state space '''
        return np.prod(self.env.observation_space.shape)

    def memorize_and_predict(self, action, state, new_state, reward, done):
        ''' Memorize the transition and train predictive models '''
        super().memorize(action, state, new_state)
        self.train_predictive_models(action, state, new_state)

    def decide_action(self, state=None) -> tuple[int, bool]:
        ''' Decide action based on predictive models or random choice '''
        # if not self.initial_training_done: # change to if models not trained yet
        #    action, new = self.new_random_step()
        #    return action, new
        unexplored = self.unused_actions()
        if len(unexplored) > 0:
            ordered = self.predictor.actions_by_expected_information_gain(
                state=state or self.state,
                unexplored_actions=unexplored)
            if len(ordered) > 0:
                return ordered[0], True
        return self.env.action_space.sample(), False

    def act(self):
        ''' Override act to incorporate decision making based on predictive models '''
        current_state = self.state
        action = self.decide_action(current_state)
        new_state, reward, done, _ = self.env.step(action)
        self.memorize_and_predict(
            action, current_state, new_state, reward, done)
        if done:
            self.env.reset()

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
        self.initial_training_done = True
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
                action, new = self.decide_action()
                # the state is saved on the environment
                _state, _reward, _done, _info = self.env.step(action)
                self.memorize()
                learnedSomethingNew = (
                    learnedSomethingNew + 1) if new else learnedSomethingNew
                self.train_predictive_models(action, self.prior, self.state)
                self.train_pathway_models(self.prior, self.state)
        return learnedSomethingNew