'''
The Naive Agent is 'naive' because it employs no AI or ML technology.

It is a simple agent that merely records all of its actions and subsequent
observations, then uses a simple bidirectional breath-first-search to find a 
path through the state space of the environment to the target state.

It can handle small environments with edges (state to state transitions) in the
order of up to maybe 10 thousand. This is because it must explore the vast 
majority of the environment in order to be generally useful as a sensorimotor
inference engine.
'''
from .agent import NaiveAgent
