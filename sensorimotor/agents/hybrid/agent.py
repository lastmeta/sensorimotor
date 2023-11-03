''' https://chat.openai.com/share/7d7c12f6-7880-4083-8a59-16e64ca85f5e '''

import numpy as np
from sensorimotor.agents.hybrid.predict.predictor import Predictor
from sensorimotor.agents.naive import NaiveAgent


class HybridAgent(NaiveAgent):
    ''' The Hybrid Agent uses predictive models to enhance naive exploration '''

    def __init__(self, env, state=None):
        super().__init__(env, state)
        action_list = list(range(self.env.action_space.n))
        self.predictor = Predictor(
            actions=action_list, dimension=self.state_dimension())
        self.initial_training_done = False

    def state_dimension(self):
        ''' Returns the dimensionality of the state space '''
        return np.prod(self.env.observation_space.shape)

    def memorize_and_predict(self, action, state, new_state, reward, done):
        ''' Memorize the transition and train predictive models '''
        super().memorize(action, state, new_state)
        self.train_predictive_models(action, state, new_state)

    def train_predictive_models(self, action, current_state, next_state):
        ''' Train the predictive models with new state transition '''
        x_train = np.array([current_state])
        y_train = np.array([next_state])
        self.predictor.train_future_action(action, x_train, y_train)
        self.predictor.train_past_action(action, y_train, x_train)

    def decide_action(self, state):
        ''' Decide action based on predictive models or random choice '''
        if not self.initial_training_done:
            action, _ = self.new_random_step()
            return action
        unexplored = self.unused_actions()
        if len(unexplored) > 0:
            ordered = self.predictor.actions_by_expected_information_gain(
                state=state,
                unexplored_actions=unexplored)
            if len(ordered) > 0:
                return ordered[0]
        return self.env.action_space.sample()

    def train(self, epocs=1, steps=1000, verbose=False, extraVerbose=False):
        ''' Extended training function that also handles predictive model training '''
        super().train(epocs, steps, verbose, extraVerbose)
        # After initial training, we can start using the predictive models
        self.initial_training_done = True

    def act(self):
        ''' Override act to incorporate decision making based on predictive models '''
        current_state = self.state
        action = self.decide_action(current_state)
        new_state, reward, done, _ = self.env.step(action)
        self.memorize_and_predict(
            action, current_state, new_state, reward, done)
        if done:
            self.env.reset()
