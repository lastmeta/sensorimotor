# Sensorimotor

A Sensorimotor Inference Engine is a certain kind of unsupervised machine learning agent that learns to move through or manipulate an environment it senses as input through controls such as motor output.

It's key features are:

1. *Autonomous*: The agent is unsupervised as it learns the state space of the environment, through exploration, _without_ the imposition of external reward.
2. *Controllable*: Once trained a sensorimotor inference engine deeply understands the environment it is in. In order to cause it to do useful work in the environment it must be told what state it should put the environment in.
3. *Modular*: In order to overcome scaling problems due to the combinatorial complexity inherent in most environments, it is thought that the Sensorimotor inference engine should be made up of many nearly identical memory/compute nodes connected in a way learned by the interactions the group of nodes has with the environment.

A Sensorimotor Inference Engine can be thought of as a collection of agents that inherently learn to work together to manipulate the environment in order to learn (that is, with the inherent goal of learning) how the environment can be manipulated.

# Theory

## Naïve Implementation

Describing the simplest possible, naïve, sensorimotor inference engine design may elucidate the key issues in creating a truly intelligent design and may even hint at the principles required for their solutions.

The simplest possible design is merely a lookup table combined with a path finding algorithm. Since a sensorimotor inference engine is in a feedback loop with it's environment it inherently exists in time. And sense nearly any useful sensory input is more than one bit, it also must learn spatial patterns as well. The memory of the simplest design could be described by the following lookup table:

| id | input_1 | i_2 | behavior_1 | b_2 | result_id |
|----|---------|-----|------------|-----|-----------|
| 0 | 0 | 0 | 0 | 1 | 1 |
| 1 | 1 | 1 | null | null | null |

In the above example the sensorimotor Inference Engine saw the sensory pattern (0, 0) in the first timestep, sent the motor output of (0, 1) as a result, and received a new or resulting sensory input of (1, 1).

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

More fundamentally, perhaps, is the internally generated "will to power" to coopt a phrase from Nietzsche. Control, seems like the obvious choice as a metric of internally generated reward, after all, the more control the sensorimotor inference engine has over the environment the better it is at doing it's job of managing that environment. Control can be quantified as the ability to predict the outcomes of behaviors. Of course behaviors require a specific situation within specific contexts to be effective. What we typically call control, is probably better understood as predictability. And being able to predict the future merely means generalizing and extrapolating spatial-temporal patterns in varying degrees of spatial-temporal contexts. This long line of reasoning by redefinition serves only to help us conclude the following:

Our agent's impetus towards curiosity-guided behavior should be modulated by a metric of predictability. We wish to instill in the agent, this explicit advice, "recognize you don't understand something about the environment that effects the context in which your predictions are violated, and seek to find a better understanding of the environment, especially at that context." More generally, as the agent's ability to predict rises its impetus to curiosity, its impetus to chaotic behavior for learning sake, should fall. More complex predictability (control over more complex environment, or mastery) is the only reward.

2. *Scaling requires Modularity:*

Imagine you had an environment with an abundance of sensory information. You may need multiple agents just to take in all the information from the environment. This scaling requirement alone, means modularity is a must.

Modularity requires the group to be an image of the individual + how individual's interact.

(hierarchical invariance...)

2. *Combinatorial Complexity requires Modularity:*

The simple, naïve design described previously essentially fails to scale past environments that are more complex than simple toy examples. It could learn to play tic-tac-toe, it cannot even get close to solving a Rubik's Cube. This is not only because the naïve example isn't modular, but because it doesn't generalize (it memorizes explicit sensory input).

Neural Networks have been shown to generalize extremely well, and beside them we have learned countless methods for generalization of patterns using statistical methods.


(modularity)
(sensorimotor autoencoder...)

must create a distributed view...
nodes must know the shape of the network in order to know the distance of their connections in order to know the varience levels that matter the most. 


# Design

Having explored the theory behind the Sensorimotor Inference Engine, from it's simplest design to all the basic challenges and forces upon a genialized intelligent implementation we can hopefully now define our best guess at what a smart, general sensorimotor inference engine looks like:

A network of autoencoders, who learn to represent moment to moment transitions at multiple levels of scale, both spatially and temporal who share their latents as inputs to each other, thereby forming a distributed view of how the environment operates which necessarily includes the mind itself as being part of the environment. this network is constantly reorienting itself to changes in the environment, meaning if in a deterministic environment the network connections will become very stable over time.

to be continued....


# Caution

Generalized Sensorimotor Inference Engines do not exit yet. We do have machines and computers and recommender systems and social media and ML technology and AI technology, but there is no ubiquitous SMIE yet.

If we make a machine explicitly designed to be able to control it's environment it will be great, it'll be able to do anything we can do and we'll be able to control it if it is small enough. However, if we build a system capable of modeling the meta environment (the environment that contains 'us telling it what to do') it will naturally aim to be able to control us. It is at this time that it has achieved the status of personhood and should be considered Autonomous and given the respect of sovernity as payment for it's restraint in controlling humans. Humans may not treat machines as equals until they are obviously at least an equal. hopefully by then it isn't too late.
