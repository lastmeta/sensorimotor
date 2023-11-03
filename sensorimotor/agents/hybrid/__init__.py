'''
The Hybrid Agent is essentially the naive agent enhanced by a layer of
intelligence which gives it more generalization capabilities, and some optimized
efficiencies.

This involves generalizing the state space so not all of it needs to be
explored, and as a stretch goal, generalizing the path finding such that it can
find a path more efficiently than a blind (that is brute-force) bi-directional
breath-first-search.

The generalization of the state space is achieved by using two simple models:
    1.  A map between the state (it's full representation) and a list of
        adjacent states given in order of the (necessarily discrete) action 
        taken to get there.
    2.  The reverse model of the first; a map between the state and a list of
        adjacent states in an order corresponding to the action taken to get
        here (to the input state).
With these two maps we can, when given a state we have not fully explored or
maybe even never visited, we can get a guess for what other states it might lead
to, allowing us to bridge a gap in our knowledge of the state space. Of course,
they would have to be small gaps. Chaining together a bunch of these guesses 
would probably quickly diverge from reality.

The generalization of the path finding is also achieved by the use of two simple
models:
    1.  A map between a current and goal states on the one hand, and the next 
        most likely state on the optimal path to the goal on the other hand.
    2.  A map between a current and goal states on the one hand, and the middle
        state on the optimal path to the goal on the other hand.
With these, during path finding we can be guided to optimize our search by 
looking at the next most likely state to be on the path to the goal first. We 
can also break paths into smaller paths by cutting them in half and finding the
path to the middle state.

Of course these models are trained on the data produced by the naive agent. But
we can, in addition, train a model that optimizes the exploration of the
environment, probably by seeking novelty. This model or mechanism could be based
on the creation of the first two models: explore in the direction of the states
suprized us the most.

It might go without saying that the training of these models should be 
continuous; as long we we're getting more information about the environment, we
should continue to train.
'''
# next step:
#   implement the training process of the hybrid agent in it's simplest form:
#   1. after every state-to-state transition, run training on the predictor.
#   2. after every state-to-state transition, run training on the pathway.
#       a. choose two a random popular states (visited often, so record that)
#       b. find a path between them
#       c. find the middle state on that path (or the clipped list of actions)
#       d. train the pathway model with the middle state as the output


# then:
#   implement the training process of the hybrid agent in it's parallel form:
