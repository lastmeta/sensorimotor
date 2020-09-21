## TODO import from envs... 

import copy
import time
import numpy as np
np.set_printoptions(precision=8, suppress=True, linewidth=400, threshold=100)
import gym


class SensorimotorAutoencoderAgents(object):
    '''
    a group of autoencoders, each with the ability to encode one transition
    that work together to form a predictive sensorimotor inference engine.
    basically they map the space, distributedly, so that they can find a path
    from any observation to any other observation - they know how to manipulate
    the environment.

    they have overlapping input bits, but no two have the same inputs. some
    have no inputs from the environment at all, and instead get inputs only
    from other autoencoders. There are typically many autoencoders. they
    automatically wire themselves up (inefficienty, but successfully).
    '''

    def __init__(self, env, encoders_n=12):
        self.env = env
        self.encoders = self.generate_encoders(encoders_n)

    def generate_encoders(self, n):
        ''' https://blog.keras.io/building-autoencoders-in-keras.html '''
        from keras.layers import Input, Dense
        from keras.models import Model
        encoders = []
        for i in range(n):
            # this is the size of our encoded representations
            encoding_dim = 4
            # this is our input placeholder
            # notice this isn't even as big as the SimpleCube Environment. (48)
            # we'll make 6 autoencoders: each one will have 20 faces they want
            # to watch. each face is one bit. so each face will be seen by 2
            # (because every case is atleast an edge case) or in some cases,
            # the corner cases, 3. In this network every node is also watching
            # one behavior: naturally, the behavior that effects the inputs it's
            # watching: and they're watching 4 friends, not 5, so they have a
            # partial view of the environment, partial view of the behaviors,
            # and a partial view of the other members of the group. Yet they
            # have more than they need to know what to do. In other words, this
            # is just the first step of the test, which could make each node way
            # more granular. we could have 1 autoencoder per face, they could
            # see one action, and we'd give them connections to many neighbors.
            # doing it granularly is not only possible but gives the whole brain
            # more ability to understand patterns through time. by the way, in
            # rendition, the latents are 8 bit representations of just the
            # autoencoders sensory input. in more granular views of the
            # enivornment you would include the neighbors representations in
            # the latents.

            input_img = Input(shape=(72,))
            # "encoded" is the encoded representation of the input
            encoded = Dense(encoding_dim, activation='relu')(input_img)
            # "decoded" is the lossy reconstruction of the input
            decoded = Dense(784, activation='sigmoid')(encoded)
            # this model maps an input to its reconstruction
            autoencoder = Model(input_img, decoded)
            # Let's also create a separate encoder model:
            # this model maps an input to its encoded representation
            encoder = Model(input_img, encoded)
            # As well as the decoder model:
            # create a placeholder for an encoded (32-dimensional) input
            encoded_input = Input(shape=(encoding_dim,))
            # retrieve the last layer of the autoencoder model
            # Here we need to change this:
            # we want the decoder_layer to be the next timestep so we can train the
            # autoencoder on the transition from
            # one observation+action to a new observation:
            decoder_layer = autoencoder.layers[-1]
            # create the decoder model
            decoder = Model(encoded_input, decoder_layer(encoded_input))
            # autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
            encoders.append(autoencoder)
        # now wire them up up so they share latents to each other's inputs (at random)
        # also wire them up at random to the environment, and the action space...
        return encoders

    def step(self, obs):
        # they are predicting what action they will take. at first the observation
        # stands in as a random seed to activate the network, but soon they
        # wire up in a hierarchy and take actions to acheive what they think
        # they will see, instead of providing goals, you provide an image of
        # what you want them to see at the top layer of the hierarchy...
        sampled = env.action_space.sample()
        print(f'action sampled: {sampled}')
        return sampled


if __name__ == '__main__':
    env = RubixCube()
    env.seed(0)
    print("agent: env.action_space {}".format(env.action_space))
    agent = SensorimotorAutoencoderAgents(env)
    for i_episode in range(1):
        obs = env.reset()
        env.render()
        for t_timesteps in range(1000):
            action = agent.step(obs)
            obs, reward, done, info = env.step(action)
    env.close()
