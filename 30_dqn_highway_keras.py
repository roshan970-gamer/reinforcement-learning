# Simple DQN example using TensorFlow/Keras.
# Install once in VS Code:
# py -m pip install tensorflow numpy

import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random

state_size, action_size = 4, 3
model = keras.Sequential([
    keras.layers.Input(shape=(state_size,)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(action_size)
])
model.compile(optimizer=keras.optimizers.Adam(0.001), loss="mse")

target = keras.models.clone_model(model)
target.set_weights(model.get_weights())
memory = deque(maxlen=2000)

gamma, epsilon = 0.95, 1.0
for episode in range(100):
    state = np.random.rand(state_size).astype(np.float32)
    total = 0

    for _ in range(30):
        action = np.random.randint(action_size) if np.random.rand() < epsilon else np.argmax(model.predict(state[None,:], verbose=0)[0])

        next_state = np.random.rand(state_size).astype(np.float32)
        reward = 1.0 if action == 1 else -0.2
        done = False
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total += reward

        if len(memory) >= 32:
            batch = random.sample(memory, 32)
            s = np.array([x[0] for x in batch])
            a = np.array([x[1] for x in batch])
            r = np.array([x[2] for x in batch])
            ns = np.array([x[3] for x in batch])

            q = model.predict(s, verbose=0)
            nq = target.predict(ns, verbose=0)
            q[np.arange(32), a] = r + gamma*np.max(nq, axis=1)
            model.fit(s, q, epochs=1, verbose=0)

    epsilon = max(0.05, epsilon*0.98)
    if episode % 10 == 0:
        target.set_weights(model.get_weights())
        print("Episode:", episode, "Reward:", round(total, 2))

print("DQN training completed.")
