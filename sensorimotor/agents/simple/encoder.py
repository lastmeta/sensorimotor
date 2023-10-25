# # # Enocder Circuit

import keras
from keras.layers import Input, Dense
from keras.models import Model
# Define the size of the encoded representations
# 32 floats -> compression factor of 24.5, assuming the input is 784 floats
encoding_dim = 32
# Define the input layer
input_img = Input(shape=(784,))
# Define the encoder layer
encoded = Dense(encoding_dim, activation='relu')(input_img)
# Define the decoder layer
decoded = Dense(784, activation='sigmoid')(encoded)
# Create the autoencoder model
autoencoder = Model(input_img, decoded)
# Compile the autoencoder model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
# Now you can train the autoencoder using your data
# Assuming `x_train` is your input data and `x_test` is your validation data
# autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True, validation_data=(x_test, x_test))
