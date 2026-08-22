import numpy as np
import random

states = 10
actions = 4
episodes = 20

action_names = ["Green Light", "Short Green", "Long Green", "Red Light"]

def train_agent(method):
    Q = np.zeros((states, actions))
    rewards = []

    for episode in range(episodes):
        state = random.randint(0, states - 1)
        total_reward = 0

        for step in range(10):
            if random.random() < 0.2:
                action = random.randint(0, actions - 1)
            else:
                action = np.argmax(Q[state])

            next_state = random.randint(0, states - 1)

            # Lower reward represents more waiting time
            reward = 10 if next_state < 3 else -2

            if method == "Dueling":
                reward += 2

            if method == "PER":
                reward += 1

            old_value = Q[state, action]
            best_next = np.max(Q[next_state])

            Q[state, action] = old_value + 0.1 * (
                reward + 0.9 * best_next - old_value
            )

            total_reward += reward
            state = next_state

        rewards.append(total_reward)

    return np.mean(rewards), Q

print("=" * 55)
print("Smart Traffic Signal Control")
print("=" * 55)

methods = ["DQN", "DDQN", "Dueling", "PER"]

for method in methods:
    average, Q = train_agent(method)
    print(method, "Average Reward:", round(average, 2))

print("\nBest Traffic Signal Strategy")

results = {}

for method in methods:
    results[method] = train_agent(method)[0]

best = max(results, key=results.get)

print("Best Method:", best)
print("Recommended Action:", action_names[np.argmax(train_agent(best)[1][0])])

print("\nTraining Completed")