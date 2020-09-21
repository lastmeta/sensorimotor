# Sensorimotor

A Sensorimotor Inference Engine is a certain kind of unsupervised machine learning agent that learns to move through or manipulate an environment it senses as input through controls such as motor output.

It's key features are:

1. *Autonomous*: The agent is unsupervised as it learns the state space of the environment, through exploration, _without_ the imposition of external reward.
2. *Controllable*: Once trained a sensorimotor inference engine deeply understands the environment it is in. In order to cause it to do useful work in the environment, it must be told what state it should put the environment in.
3. *Modular*: In order to overcome scaling problems due to the combinatorial complexity inherent in most environments, it is thought that the Sensorimotor inference engine should be made up of many nearly identical memory/compute nodes connected in a way learned by the interactions the group of nodes has with the environment.

A Sensorimotor Inference Engine can be thought of as a collection of agents that inherently learn to work together to manipulate the environment in order to learn (that is, with the inherent goal of learning) how the environment can be manipulated.

# Theory

A sensorimotor inference engine is a simple concept - it's a machine that learns how to interact with something (called the environment) so you don't have to. You merely let the machine explore the environment on its own, then you can tell the machine to put the environment in any particular state you wish and it will carry out an efficient set of behaviors to accomplish your goal. A Sensorimotor Inference Engine allows you to abstract entire systems away.

Though this is a simple idea, it may be helpful to discuss the shortcomings of its simplest implementation in order to understand the requirements needed to design a true, generally intelligent Sensorimotor Inference Engine.

## Naïve Implementation

Describing the simplest possible, naïve, sensorimotor inference engine design may elucidate the key issues in creating a truly intelligent design and may even hint at the principles required for their solutions.

The simplest possible design is merely a lookup table combined with a pathfinding algorithm. Since a sensorimotor inference engine is in a feedback loop with it's environment it inherently exists in time. And sense nearly any useful sensory input is more than one bit, it also must learn spatial patterns as well. The memory of the simplest design could be described by the following lookup table:

| id | input_1 | input_2 | behavior_1 | behavior_2 | result_id |
|----|---------|-----|------------|-----|-----------|
| 0 | 0 | 0 | 0 | 1 | 1 |
| 1 | 1 | 1 | null | null | null |

In the above example, the sensorimotor Inference Engine saw the sensory pattern (0, 0) in the first timestep, sent the motor output of (0, 1) as a result, and received a new or resulting sensory input of (1, 1).

A memory structure of this design could be traversed by a pathfinding algorithm such that given any previously seen input, it could look for a path to any previously seen desired input. Just by executing the behaviors listed in this found path, the engine would travel the state space to manipulate the environment into conforming to the desired configuration.

This example serves only to communicate precisely what a sensorimotor inference engine is because it is entirely useless in the real world. As any ml or ai novice knows the problems faced in AI are about overcoming the problems that this design would face, mainly: it doesn't generalize to input patterns it hasn't seen. There's no _intelligence_ in this design; it's just explicit memory + a static computational algorithm.

Still, in very simple environments where not only the entire state space but also the entire transition space can be explored through random behavior in a short amount of time, this naïve design is a functioning sensorimotor inference engine because it can train itself *autonomously* (in this case through random behavior) and it can be *controlled*.

Therefore this simplistic implementation can serve as an example of what an intelligent sensorimotor inference engine must be.

## Intelligent Implementation

The entire industry of ML and AI exists to overcome the problems of combinatorial complexity and generalization faced by the naïve implementation outlined above. Therefore, in attempting to implement a 'smart' version of the above design, nothing radically new needs to be developed.

As an example, Generative Adversarial Networks were originally conceived as a new way to combine already existing technology (certain types of neural networks) in a novel way to create a path of circular training, and exponentially advance the quality existing technology could produce. Of course, Generative Adversarial Networks introduced new, nuanced problems requiring their new solutions, but generally, Generative Adversarial Networks represent merely a new mashup of preexisting technology.

In much the same way Sensorimotor Inference Engines require no radically new technology to be developed, but will inevitably introduce new, unique challenges in their development to scale.

### Aim

The goal of this project, embodied in this repository, is to discover and implement a minimal viable design that produces intelligent manipulation of an environment generally and thus in combination conform to the key features which define a Sensorimotor Inference Engine.

### Challenges

There are at least 3 conceptual challenges to creating a Sensorimotor Inference Engine:

1. *No External Rewards while Training:*

Today's intelligent technology that traverse environments typically learns to do so by training the agent with a reward. This creates an agent that doesn't understand the environment itself, but instead, understands the reward schema in relation to the environment. Since once the agent is trained it may be told to manipulate the environment to any end, the agent must not be biased by an external reward system.

Instead, if 'rewards' are required for training at all the rewards must be fundamentally internal. For example, some work has been done on curiosity and attention training, some solutions may be found there.

More fundamentally, perhaps, is the internally generated "will to power" to borrow a phrase from Nietzsche. Control, seems like the obvious choice as a metric of internally generated reward, after all, the more control the sensorimotor inference engine has over the environment the better it is at doing its job of managing that environment. Control can be quantified as the ability to predict the outcomes of behaviors. Of course, behaviors require a specific situation within specific contexts to be effective. What we typically call control, is probably better understood as predictability. And being able to predict the future merely means generalizing and extrapolating spatial-temporal patterns in varying degrees of spatial-temporal contexts. This long line of reasoning by redefinition serves only to help us conclude the following:

Our agent's impetus towards curiosity-guided behavior should be modulated by a metric of predictability. We wish to instill in the agent, this explicit advice, "recognize you don't understand something about the environment that affects the context in which your predictions are violated, and seek to find a better understanding of the environment, especially at that context." More generally, as the agent's ability to predict rises its impetus to curiosity, its impetus to chaotic behavior for learning's sake should fall. More complex predictability (control over a more complex environment, or mastery) is the only reward.

2. *Must Scale and Must Generalize:*

The simple, naïve design described previously essentially fails to scale past environments that are more complex than simple toy examples. It could learn to play tic-tac-toe, it cannot even get close to solving a Rubik's Cube. This is because the naïve example doesn't generalize (it memorizes explicit sensory input).

Some environments have a large input space. For instance, the human body has billions of receptors acting as sensory input for the brain. The only way to handle such an environment is to have multiple nodes of perception which further communicate with each other. In other words, modularity is a requirement for scale.

Thus a what is needed is a network of sensorimotor agents, communicating and working together to control the environment. Each node in the network, each agent should be nearly identical and doing essentially the same thing as all other nodes in the network but have a unique set of connections so as to give it a unique view of the environment and of the group. It connections should overlap with others though.

These nodes that each see an overlapping portion of the sensory input stream should be able to communicate with each other to define a combined, distributed view of the environment. They must generalize the patterns they see instead of explicitly memorizing them, and they must constantly predict the future of their own input streams.

3. *Context Matters:*

Seeing a pattern in one context can mean something very different from seeing the same pattern in a different context. A good proxy for context is often the past. Of course, the immediate past lives within its own context, the less immediate past, and so on to the distant past. A generally intelligent sensorimotor inference engine must know what motor commands to send given any set of contexts.

The way to deal with layered contextual data is to develop a hierarchy that understands or embodies the contextual hierarchy. A hierarchy, like a pyramid, has a large base, and small top. highly invariant patterns are known by the nodes at the top of the hierarchy, just as the CEO of a company plans for the long term. The multitude of nodes at the bottom of the hierarchy plan for and understand in great detail some particular aspect of a short term feedback loop, just as employees of a company manage day to day concerns and know-how to carry out very particular tasks.

Hierarchy is such a ubiquitous concept in any organized informational structure that our network of sensorimotor agents must be able to form hierarchies in order to efficiently mirror and manage layered spatial-temporal contexts.

# Conceptual Design

Having explored the theory behind the Sensorimotor Inference Engine, from its simplest design to the basic challenges faced by a generalized intelligent implementation we can hopefully now define our best guess at what a smart, general sensorimotor inference engine looks like.

Neural Networks have been shown to generalize extremely well. Luckily for us, there is a simple neural network called an autoencoder that fits many of the other requirements outlined above as well. An autoencoder produces a compressed image of its input. This is essential since in order to communicate with other autoencoders we need a smaller representation. Moreover, the autoencoder can be made to encode the transition from one sensory-input to the next, making it a predictor of the future of any sensory input.

If arranged in a hierarchy the mere prediction of top-level nodes actually serves to bias and inform the behavior of lower-level nodes: turning that prediction into a self-fulfilling prophecy. Indeed, however, the top-level nodes receive most of their information about what is going on in the environment as the aggregated summaries of lower-level nodes, thus in aggregate, the lower level nodes are able to affect the interpretation of the environment, effectively exerting influence over the entire network.

Indeed, though we can't make everything we need to out of this one type of neural network, it seems we're developing the idea of a *sensorimotor autoencoder* (which is actually a group of autoencoders working together to predict and thereby determine the future of the environment they interact with).

What informs these connections between nodes? The implementation may be very complex but the question has already been answered - whatever connections allow a node to predict the future of its input stream better, are the connections it forms.

## What about the Hierarchy?

We have a network in mind, a network of nearly identical agents. Hierarchy can seem, at first glance, antithetical to networks.

Fortunately, there is a way that hierarchy can fall out of a flat network all on its own. In a network that is not fully connected connections can be given a distance metric. This metric is essentially how closely interconnected the nodes are. As an example, do I have a friend that knows nobody else I know? Even if that friend lives in the same city as me, they are a more 'distant' connection than my other friends who all know each other.

If, in the sensorimotor autoencoder network, we were able to identify distant connections we could listen to them at the appropriate rate of invariance. A distant connection, one that receives very different information about the environment can serve as a useful low-resolution flag or indicator, while nodes that see some of the same inputs and are very similar can share higher resolution data because they all are familiar with the same patterns, the same context: they can talk shop with each other.

In order to achieve this, we need two things. First of all, each autoencoder (that is each node) must actually be several autoencoders, encoding the input stream at various rates of compression. The smaller representations must be shared with 'further away' connections, the larger, more detailed, less compressed representations must be shared with 'nearby' connections. Secondly, the nodes themselves must have a map of the network of nodes: they must know the shape of the network in order to know the distance of their connections... in order to know the correct variance levels to share. This shared map of connections shouldn't be difficult to define and distribute, but it is something in each node that stands apart from the neural net autoencoder.

## Conceptual Design Overview

What we are left with are a network of autoencoders, who learn to represent moment to moment transitions at multiple levels of scale, both spatially and temporal who share their compressed representations as inputs to each other, thereby forming a distributed view of how the environment operates which necessarily includes the group mind, itself, as being part of the environment. Each node in the network stands-in at various levels of multiple hierarchies as depicted by its particular set of learned connections. This network is constantly reorienting itself to changes in the environment, meaning if in a deterministic environment the network connections will become very stable over time.

# Development Roadmap

Conceivably the best way to develop and test the above design is to use a series of environments of increasing complexity. For instance, if a very simple environment can be managed by such a system to our satisfaction, then a more complex environment can be tested.

We might start out with a number line, or some simple state space with few (linear) transitional steps, then move onto a 2d space with obstacles, finally, a Rubik's Cube may be sufficiently large a state space and transition space to satisfy all requirements, at least as far as deterministic environments are concerned. Non-deterministic environments should be able to be managed by a sufficiently intelligent Sensorimotor Inference Engine, however, the aim of this repository is first and foremost the development of a minimal viable general Sensorimotor Inference Engine, so non-deterministic environments, such as any multiplayer game (since another player's actions are essentially undetermined) is beyond our scope at this time.

OpenAI Gym https://gym.openai.com/ seems to serve as an ideal framework by which to develop environments that our network of agents can interact with.

to be continued...

---

# A Word of Caution

Generalized Sensorimotor Inference Engines do not exist yet. We do have machines and computers and recommender systems and social media and ML technology and AI technology, but there is no ubiquitous general-purpose Sensorimotor Inference Engine yet.

If we make a machine explicitly designed to be able to control its environment we will be able to abstract many concerns away, and abundance will be the inevitable result. However, if we build a system capable of modeling (and if it is capable of modeling this, it will inevitably do so) the meta environment (the environment that contains "us telling it what to do") it will naturally aim to be able to control that environment as well. That seems like a precarious situation for the living.
