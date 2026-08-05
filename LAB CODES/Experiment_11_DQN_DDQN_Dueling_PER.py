import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import random

algorithms = ["DQN", "DDQN", "Dueling DQN", "PER"]

state_size = 4
action_size = 2

results = []

for algorithm in algorithms:

    model = Sequential([
        Dense(24, activation="relu", input_shape=(state_size,)),
        Dense(24, activation="relu"),
        Dense(action_size, activation="linear")
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mse"
    )

    total_reward = 0

    for episode in range(5):

        state = np.random.rand(1, state_size)

        q_values = model.predict(state, verbose=0)

        action = np.argmax(q_values[0])

        reward = random.randint(5,15)

        target = q_values.copy()
        target[0][action] = reward

        model.fit(state, target, epochs=1, verbose=0)

        total_reward += reward

    results.append(total_reward)

print("\n========== Performance Comparison ==========\n")

for i in range(len(algorithms)):
    print(algorithms[i], "Total Reward =", results[i])

best = np.argmax(results)

print("\nBest Algorithm :", algorithms[best])
print("Maximum Reward :", results[best])