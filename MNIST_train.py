#testing code with established dataset

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess the data
X_train = X_train.reshape(X_train.shape[0], 28, 28) / 255.0
X_test = X_test.reshape(X_test.shape[0], 28, 28) / 255.0

# Define the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(28, 28)))  # Input shape is (time steps, features)
model.add(Dense(10, activation='softmax'))  # 10 output classes for digits 0-9

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)

print(f"Test accuracy: {accuracy*100:.2f}%")
