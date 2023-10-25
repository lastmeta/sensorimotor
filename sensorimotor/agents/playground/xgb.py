# import time
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder

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

# Convert y_train to label-encoded instead of one-hot encoded
y_train_label_encoded = np.argmax(y_train, axis=1)

# Now, x_train_repeated and y_train_repeated have more instances, repeated.

# Initialize and train the XGBoost model
model = xgb.XGBClassifier(objective="multi:softmax",
                          num_class=4, n_estimators=100)
# model.fit(x_train, y_train_label_encoded)
n_repeat = 10  # Number of times to repeat
x_train_repeated = np.tile(x_train, (n_repeat, 1))
y_train_repeated = np.tile(y_train_label_encoded, n_repeat)
# start = time.time()
model.fit(x_train_repeated, y_train_repeated)

# Make predictions
predictions = model.predict(x_train)

# Convert the label-encoded predictions back to one-hot encoding
encoder = OneHotEncoder(sparse_output=False, categories='auto')
predictions_one_hot = encoder.fit_transform(predictions.reshape(-1, 1))

# Check if mapping is learned
if np.array_equal(predictions, y_train_label_encoded):
    print("Mapping learned.", predictions, y_train_label_encoded)
else:
    print("Mapping not learned.", predictions, y_train_label_encoded)

# end = time.time()
# print(f"Time elapsed: {end - start} seconds.") Time elapsed: 0.1591794490814209 seconds.
