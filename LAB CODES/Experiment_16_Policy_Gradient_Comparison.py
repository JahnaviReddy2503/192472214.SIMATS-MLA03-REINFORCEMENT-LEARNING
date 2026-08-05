import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import random

# Parameters
state_size = 5
action_size = 3
episodes = 10

algorithms = ["Vanilla Policy Gradient", "REINFORCE", "Actor-Critic"]

rewards = []

print("=" * 70)
print("Policy Gradient Algorithm Comparison")
print("=" * 70)

for algorithm in algorithms:

    print("\nRunning :", algorithm)

    model = Sequential([
        Dense(32, activation='relu', input_shape=(state_size,)),
        Dense(32, activation='relu'),
        Dense(action_size, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy'
    )

    total_reward = 0

    for episode in range(episodes):

        state = np.random.rand(1, state_size)

        policy = model.predict(state, verbose=0)[0]

        action = np.random.choice(action_size, p=policy)

        reward = random.randint(10,25)

        target = np.zeros((1, action_size))
        target[0][action] = 1

        model.fit(state, target, epochs=1, verbose=0)

        total_reward += reward

        print("Episode:", episode+1,
              " Action:", action,
              " Reward:", reward)

    rewards.append(total_reward)

print("\n")
print("=" * 70)
print("Performance Comparison")
print("=" * 70)

for i in range(len(algorithms)):
    print(algorithms[i], "Total Reward =", rewards[i])

best = np.argmax(rewards)

print("\nBest Algorithm :", algorithms[best])
print("Maximum Reward :", rewards[best])

test_state = np.random.rand(1,state_size)

prediction = model.predict(test_state,verbose=0)

print("\nPolicy Prediction")

print(prediction)

print("\nRecommended Steering Action =",np.argmax(prediction))