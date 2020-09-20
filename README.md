# Sensorimotor

A Sensorimotor Inference Engine is a certain kind of unsupervised machine learning agent that learns to move through or manipulate an environment it senses as input through controls such as motor output.

It's key features are:

1. *Autonomous*: The agent is unsupervised as it learns the state space of the environment, through exploration, _without_ the imposition of external reward.
2. *Modular*: In order to overcome scaling problems due to the combinatorial complexity inherent in most environments, it is thought that the Sensorimotor inference engine should be made up of many nearly identical memory/compute nodes connected in a way learned by the interactions the group of nodes has with the environment.
3. *Controllable*: Once trained a sensorimotor inference engine deeply understands the environment it is in. In order to cause it to do useful work in the environment it must be told what state it should put the environment in.

A Sensorimotor Inference Engine can be thought of as a collection of agents that inherently learn to work together to manipulate the environment in order to learn (that is, with the inherent goal of learning) how the environment can be manipulated.

# Theory

## Naïve Implementation

Describing the simplest possible, naïve, sensorimotor inference engine design may elucidate the key issues in creating a truly intelligent design and may even hint at the principles required for their solutions.

The simplest possible design is merely a lookup table combined with a path finding algorithm. Since a sensorimotor inference engine is in a feedback loop with it's environment it inherently exists in time. And sense nearly any useful sensory input is more than one bit, it also must learn spatial patterns as well. The memory of the simplest design could be described by the following lookup table:

| id | input_1 | i_2 | behavior_1 | b_2 | result_id |
|----|---------|-----|------------|-----|-----------|
| 0 | 0 | 0 | 0 | 1 | 1 |
| 1 | 1 | 1 | null | null | null |

In the above example the sensorimotor Inference Engine saw the sensory pattern (0, 0, ...) in the first timestep, sent the motor output of (0,1, ...) as a result, and received a new or resulting sensory input of (1,1...).

A memory structure of this design could be traversed by a path finding algorithm such that given any previously seen input, it could look for a path to any previously seen desired input. Just by executing the behaviors listed in this found path, the engine would travel the state space to manipulate the environment into conforming to the desired configuration.

This example serves only to communicate precisely what a sensorimotor inference engine is, because it is entirely useless in the real world. As any ml or ai novice knows the problems faced in AI are about overcoming the problems that this design would face, mainly: it doesn't generalize to input patterns it hasn't seen. There's no _intelligence_ in this design; it's just explicit memory + a static computational algorithm.

Still in very simple environments where not only the entire state space, but also the entire transition space can be explored through random behavior in a short amount of time, this naïve design is a functioning sensorimotor inference engine because it can train itself *autonomously* (in this case through random behavior) and it can be *controlled*.

Therefore this simplistic implementation can serve as an example of what an intelligent sensorimotor inference engine must be.

## Intelligent Implementation

The entire industry of ML and AI exists to overcome the problems of combinatorial complexity and generalization faced by the naïve implementation outlined above. Therefore, in attempting to implement a 'smart' version of the above design, nothing radically new needs to be developed.

As an example Generative Adversarial Networks were originally conceived as a new way to combine already existing technology (certain types of neural networks) in a novel way to create a path of circular training, and exponentially advance the quality exiting technology could produce. Of course Generative Adversarial Networks introduced new, nuanced problems requiring their new solutions, but generally, Generative Adversarial Networks represent merely a new mashup of preexisting technology.

In much the same way Sensorimotor Inference Engines require no radically new technology to be developed, but will inevitably introduce new, unique challenges in their development to scale.

### Aim

The goal of this project, embodied in this repository, is to discover and implement a minimal viable design which produce intelligent manipulation of an environment generally and thus in combination conform to the key features which define a Sensorimotor Inference Engine.

### Challenges and Possible Avenues for Solutions

There are at least 3 conceptual challenges to creating a Sensorimotor Inference Engine:

1. *No External Rewards while Training:*

Today's intelligent technology that manages environments typically do so by training the agent with a reward. This creates an agent that doesn't understand the environment itself, but instead, understands the reward schema in relation to the environment. Since, once the agent is trained it may be told to manipulate the environment to any end, the agent must not be biased by an external reward system.

Instead, if 'rewards' are required for training at all the rewards must be fundamentally internal. For example, some work has been done on curiosity and attention training, some solutions may be found there.

More fundamentally, perhaps, is the internally generated "will to power" to coopt a phrase from Nietzsche. Control, seems like the obvious choice as a metric of internally generated reward. Control can be quantified as the ability to predict the outcomes of behaviors, which requires the ability to predict the natural path of things (with no behavioral alteration). Thus it is fundamentally nothing more than pattern recognition through time (recognition of spatial patterns in the input space and patterns of spatial patterns through time).

to be continued...


(modularity)
(hierarchical invariance...)
(sensorimotor autoencoder...)
(...)

....


# Caution

Generalized Sensorimotor Inference Engines do not exit yet. We do have machines and computers and recommender systems and social media and ML technology and AI technology, but there is no ubiquitous SMIE yet.

If we make a machine explicitly designed to be able to control it's environment it will be great, it'll be able to do anything we can do and we'll be able to control it if it is small enough. However, if we build a system capable of modeling the meta environment (the environment that contains 'us telling it what to do') it will naturally aim to be able to control us. It is at this time that it has achieved the status of personhood and should be considered Autonomous and given the respect of sovernity as payment for it's restraint in controlling humans. Humans may not treat machines as equals until they are obviously at least an equal. hopefully by then it isn't too late.
