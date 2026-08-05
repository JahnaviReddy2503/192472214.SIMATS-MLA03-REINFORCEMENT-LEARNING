import numpy as np
import random

# Parameters
num_agents = 2
num_states = 5
num_actions = 4

alpha = 0.1
gamma = 0.9
epsilon = 0.2

episodes = 10

# Q-Tables for each agent
Q_tables = [np.zeros((num_states, num_actions)) for _ in range(num_agents)]

print("=" * 70)
print("Multi-Agent Reinforcement Learning")
print("=" * 70)

for episode in range(episodes):

    print("\nEpisode", episode + 1)

    total_reward = 0

    for agent in range(num_agents):

        state = random.randint(0, num_states - 1)

        # Epsilon-Greedy
        if random.random() < epsilon:
            action = random.randint(0, num_actions - 1)
        else:
            action = np.argmax(Q_tables[agent][state])

        reward = random.randint(5, 20)

        next_state = random.randint(0, num_states - 1)

        old_value = Q_tables[agent][state][action]

        next_max = np.max(Q_tables[agent][next_state])

        new_value = old_value + alpha * (
            reward + gamma * next_max - old_value
        )

        Q_tables[agent][state][action] = new_value

        total_reward += reward

        print("Agent :", agent + 1)
        print(" State :", state)
        print(" Action :", action)
        print(" Reward :", reward)
        print(" Updated Q Value :", round(new_value, 2))
        print()

    print("Total Episode Reward :", total_reward)

print("\n" + "=" * 70)
print("Training Completed")
print("=" * 70)

for agent in range(num_agents):

    print("\nQ Table of Agent", agent + 1)

    print(np.round(Q_tables[agent], 2))

print("\nBest Actions")

for agent in range(num_agents):

    print(
        "Agent",
        agent + 1,
        ":",

        np.argmax(Q_tables[agent], axis=1)
    )