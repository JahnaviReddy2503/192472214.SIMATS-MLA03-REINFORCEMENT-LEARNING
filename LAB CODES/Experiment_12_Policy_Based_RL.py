import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import random

# Parameters
state_size = 4
action_size = 3
episodes = 10

# Build Policy Network
model = Sequential([
    Dense(32, activation="relu", input_shape=(state_size,)),
    Dense(32, activation="relu"),
    Dense(action_size, activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy"
)

print("=" * 55)
print("Policy-Based Reinforcement Learning")
print("=" * 55)

for episode in range(episodes):

    state = np.random.rand(1, state_size)

    # Action probabilities
    probabilities = model.predict(state, verbose=0)[0]

    action = np.random.choice(action_size, p=probabilities)

    reward = random.randint(5, 20)

    target = np.zeros((1, action_size))
    target[0][action] = 1

    model.fit(state, target, epochs=1, verbose=0)

    print("---------------------------------------")
    print("Episode :", episode + 1)
    print("State :", state.round(2))
    print("Action Selected :", action)
    print("Reward :", reward)

print("\nTraining Completed Successfully")

print("\nTesting Policy")

test_state = np.random.rand(1, state_size)

policy = model.predict(test_state, verbose=0)

print("Policy Probabilities")

print(policy)

print("Best Action =", np.argmax(policy))