# import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# One-hot encoded input and output pairs
x_train = np.array([
    [1, 0, 0, 0],  # 00
    [0, 1, 0, 0],  # 01
    [0, 0, 1, 0],  # 10
    [0, 0, 0, 1],  # 11
])

y_train = np.array([
    [0, 1, 0, 0],  # 01
    [0, 0, 1, 0],  # 10
    [0, 0, 0, 1],  # 11
    [1, 0, 0, 0],  # 00
])

# Build the model
model = Sequential([
    Dense(4, input_dim=4, activation='softmax')  # Output layer
])

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
# start = time.time()
# Train, test, and potentially stop
for its in range(10):  # 1000 is the max number of its
    model.fit(x_train, y_train, epochs=100)
    predictions = model.predict(x_train)
    decoded_predictions = np.argmax(predictions, axis=1)
    if np.array_equal(decoded_predictions, np.argmax(y_train, axis=1)):
        print(f"Mapping learned after {its+1} epochs.")
        break

# end = time.time()
# print(f"Time elapsed: {end - start} seconds.") # 2.745863914489746 seconds.
