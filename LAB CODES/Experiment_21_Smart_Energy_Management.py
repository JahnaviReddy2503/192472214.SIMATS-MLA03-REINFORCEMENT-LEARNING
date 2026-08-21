import numpy as np
import random

states = 5
actions = 3

# Actions: 0 = Reduce, 1 = Maintain, 2 = Increase
action_names = ["Reduce Load", "Maintain Load", "Increase Load"]

Q = np.zeros((states, actions))

alpha = 0.1
gamma = 0.9
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.95

episodes = 100
energy_limit = 100

for episode in range(episodes):

    state = random.randint(0, states - 1)
    energy = random.randint(40, 90)

    for step in range(20):

        if random.random() < epsilon:
            action = random.randint(0, actions - 1)
        else:
            action = np.argmax(Q[state])

        if action == 0:
            energy -= 10
        elif action == 1:
            energy -= 2
        else:
            energy += 8

        energy = max(0, energy)

        if energy <= energy_limit:
            reward = 10 - energy * 0.05
        else:
            reward = -20

        next_state = min(4, max(0, int(energy / 20)))

        Q[state, action] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )

        state = next_state

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

print("=" * 55)
print("SMART ENERGY MANAGEMENT USING Q-LEARNING")
print("=" * 55)

print("\nLearned Q-Table:")
print(np.round(Q, 2))

test_state = 3
best_action = np.argmax(Q[test_state])

print("\nTest Energy State:", test_state)
print("Recommended Action:", action_names[best_action])
print("Energy Limit:", energy_limit)

print("\nResult: Energy management policy learned successfully.")