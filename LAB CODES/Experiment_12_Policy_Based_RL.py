import numpy as np
import random

states = 5
actions = 3

action_names = [
    "Move Left",
    "Move Right",
    "Pick Object"
]

policy = np.zeros((states, actions))

episodes = 20

print("Policy-Based Reinforcement Learning")
print("Industrial Robotic Arm\n")

for episode in range(1, episodes + 1):

    state = random.randint(0, states - 1)
    total_reward = 0

    for step in range(5):

        probabilities = np.exp(policy[state])
        probabilities /= np.sum(probabilities)

        action = np.random.choice(actions, p=probabilities)

        if action == 2:
            reward = 10
        else:
            reward = -1

        policy[state, action] += 0.05 * reward

        total_reward += reward

        state = random.randint(0, states - 1)

    if episode % 5 == 0:
        print("Episode:", episode,
              "Total Reward:", total_reward)

print("\nLearned Policy")

for state in range(states):
    action = np.argmax(policy[state])
    print("State", state, "->", action_names[action])

print("\nTraining Completed")