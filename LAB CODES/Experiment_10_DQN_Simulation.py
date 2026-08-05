import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import random

# Hyperparameters
state_size = 4
action_size = 2
learning_rate = 0.001
gamma = 0.95
episodes = 10

# Build DQN Model
model = Sequential([
    Dense(24, input_shape=(state_size,), activation='relu'),
    Dense(24, activation='relu'),
    Dense(action_size, activation='linear')
])

model.compile(
    loss='mse',
    optimizer=Adam(learning_rate=learning_rate)
)

print("========== Deep Q Network ==========")
model.summary()

print("\nTraining Started...\n")

for episode in range(episodes):

    # Random initial state
    state = np.random.rand(1, state_size)

    total_reward = 0

    for step in range(5):

        # Predict Q-values
        q_values = model.predict(state, verbose=0)

        # Epsilon-Greedy Action
        epsilon = 0.2

        if random.random() < epsilon:
            action = random.randint(0, action_size - 1)
        else:
            action = np.argmax(q_values[0])

        reward = random.randint(1, 10)

        next_state = np.random.rand(1, state_size)

        target = reward + gamma * np.max(model.predict(next_state, verbose=0))

        target_f = q_values.copy()
        target_f[0][action] = target

        model.fit(state, target_f, epochs=1, verbose=0)

        state = next_state

        total_reward += reward

    print(f"Episode {episode+1}  Reward = {total_reward}")

print("\nTraining Completed Successfully.")

# Test Prediction

test_state = np.random.rand(1, state_size)

prediction = model.predict(test_state, verbose=0)

print("\nPredicted Q Values")

print(prediction)

print("\nBest Action =", np.argmax(prediction))