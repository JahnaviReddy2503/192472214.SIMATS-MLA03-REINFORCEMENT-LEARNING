import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import random

# Parameters
state_size = 4
action_size = 3
tasks = ["Assembly", "Welding", "Painting"]
episodes = 10

# Meta Policy Network
model = Sequential([
    Dense(32, activation='relu', input_shape=(state_size,)),
    Dense(32, activation='relu'),
    Dense(action_size, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy'
)

print("="*70)
print("Meta Reinforcement Learning for Adaptive Industrial Robot")
print("="*70)

task_rewards = {}

for episode in range(episodes):

    task = random.choice(tasks)

    state = np.random.rand(1, state_size)

    policy = model.predict(state, verbose=0)[0]

    action = np.random.choice(action_size, p=policy)

    reward = random.randint(10,25)

    target = np.zeros((1, action_size))
    target[0][action] = 1

    model.fit(state, target, epochs=1, verbose=0)

    if task not in task_rewards:
        task_rewards[task] = []

    task_rewards[task].append(reward)

    print("\nEpisode :", episode+1)
    print("Task :", task)
    print("State :", state.round(2))
    print("Selected Action :", action)
    print("Reward :", reward)

print("\n" + "="*70)
print("Average Reward for Each Task")
print("="*70)

for task in task_rewards:
    avg = np.mean(task_rewards[task])
    print(task, ":", round(avg,2))

test_state = np.random.rand(1,state_size)

prediction = model.predict(test_state, verbose=0)

print("\nLearned Policy")

print(prediction)

print("\nBest Action =", np.argmax(prediction))