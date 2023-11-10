from itertools import combinations


def powerSet(string):
    chars = list(string)
    result = ['']  # start with the empty set
    for i in range(1, len(chars) + 1):
        for combo in combinations(chars, i):
            result.append(''.join(combo))
    return result


class EducationalModel:
    ''' 
    a simple way of building and expressing a model for educational purposes.
    our goal is to work with very simple datasets to understand exactly how 
    models are made. We would like to create a model that expresses the data 
    perfectly: generalizes 'perfectly' according to the data (of course perfect
    generalization requires perfect knowledge of the context in which the data
    was generated, which is not our goal, it is our goal to reflect the 
    generalization implied by the data itself perfectly), and perfect fit (that
    is, at each iteration of model training we always have a model that will 
    perfectly predict all prior learned observations).

    it is ok if this model is not efficient or scalable. we merely wish to 
    explicitly see how to build the perfect model on small datasets.
    '''

    def __init__(self):
        ''' 
        init the required data structures.

        we're using an input output map to make this model. the key is input 
        index to output index, and the value is the count that this input has
        lead to that output. we also need to keep all possible combinations of
        inputs...
            inputs  combos
                    null
            A       A
            B       B
            C       C 
                    AB
                    AC
                    BC
                    ABC

        datastructure: // in a non-deterministic environment the values are lists.
            {
                A: A, // identity function
                B: C, // B -> C
                C: B, // C -> B
                BC: ABC, // BC -> ABC
            }

        yes it would be nicer if we could just save neurons to neurons, rather 
        than track their combinations. but we would need an additional way to
        weight their connections accorinding to the neighbors they have.

        when we get a new combination we add up all the counts we've seen that 
        share any inputs and that is the prediction.
        '''
        self.model = {}

    def train(self, X, y):
        '''
        here we train on 1 new observation and it's result. So we learn the 
        mapping X -> y and add this mapping to our model somehow. we must do it
        in such a way that we don't mess up any previously learned mappings.

        for instance if we were to have some kind of neural network-like 
        structure as our model datastrcture we would get some kind of error 
        between what the network would currently predict for X and what y is. 
        this error implies a set of possible sets of weights we could replace 
        our current weights with to reduce the error. of that set there are two 
        important subsets that overlap: firstly there is the subset of weight 
        combinations that would reduce the error to 0, and secondly there is a
        subset of weight combinations that will not cause any problems 
        concerning the previously learned mappings. these two sets overlap and
        we want to choose a combination of weights from the intersection of the
        two sets.

        how to easily deduce what those sets are depends on how the models is 
        expressed and how we go about creating it (in much the same way as 
        building code that isn't easily broken implies certain design 
        principles and patterns).
        '''

    def predict(self, X):
        ''' 
        returns y for a given X. here we use the model we've created. 

        when we predict something we first take our most obvious assumption,
        something like things that look alike are alike and therefore produce
        alike looking results. Then when that assumption is violated we say,
        except for this case, and that case and that case, assume this instead.
        eventually we might even say, switch the basic assumption to this other
        assumption which seems to be the rule in most cases.

        layers and layers of assumptions, each covering a smaller and smaller
        group of mappings. So what is that assumption space look like, what 
        should be our default priors to begin with? decision trees have no 
        priors but we do. These assumptions are functions. The default 
        assumption as the base layer is the identity function because if we've 
        seen nothing we have no reason to give anything else back.

        when identity is violated it will be violated in a specific way. we need 
        to learn how to describe the space (controt the space) so that we can
        bring all the observations that violate the identity function into a
        proximity with each other.

        the model is an interaction between two spaces. A way to re-arrange all
        inputs in relation to each ohter, and a way that their relative 
        locations affect the outputs they produce. you want the mappings are are
        in balance.

            Identity 
        '''
        for x in powerSet(X):
            pass


# to make things simple we'll require inputs and outputs to be of the same shape
model = EducationalModel()
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [[1, 0], [0, 0], [0, 0], [1, 0]]
model.train(X[0], y[0])
model.train(X[1], y[1])
model.train(X[2], y[2])
model.train(X[3], y[3])
print(model.predict(X[0]))  # should return y[0]
print(model.predict(X[1]))  # should return y[1]
print(model.predict(X[2]))  # should return y[2]
print(model.predict(X[3]))  # should return y[3]
