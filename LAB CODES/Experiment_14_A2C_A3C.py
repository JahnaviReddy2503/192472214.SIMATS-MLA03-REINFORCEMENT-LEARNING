import numpy as np
import random

states = 5
actions = 3

action_names = [
    "Move Down",
    "Stay",
    "Move Up"
]

value = np.zeros(states)
policy = np.zeros((states, actions))

episodes = 20

print("A2C and A3C Smart Elevator Scheduling\n")

for episode in range(1, episodes + 1):

    state = random.randint(0, states - 1)
    total_reward = 0

    for step in range(5):

        probabilities = np.exp(policy[state] - np.max(policy[state]))
        probabilities /= np.sum(probabilities)

        action = np.random.choice(actions, p=probabilities)

        if action == 1:
            reward = 8
        else:
            reward = -2

        next_state = random.randint(0, states - 1)

        target = reward + 0.9 * value[next_state]

        advantage = target - value[state]

        # Critic update
        value[state] += 0.1 * advantage

        # Actor update
        policy[state, action] += 0.05 * advantage

        total_reward += reward
        state = next_state

    if episode % 5 == 0:
        print("Episode:", episode,
              "Reward:", total_reward)

print("\nLearned Elevator Policy")

for state in range(states):
    action = np.argmax(policy[state])
    print("State", state, "->", action_names[action])

print("\nA2C/A3C Training Completed")