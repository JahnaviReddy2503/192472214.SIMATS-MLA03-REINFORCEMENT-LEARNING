import numpy as np
import random

states = 5
actions = 3

action_names = [
    "Step Left",
    "Step Forward",
    "Step Right"
]

policy = np.zeros((states, actions))

episodes = 20

print("PPO and TRPO Humanoid Walking\n")

for episode in range(1, episodes + 1):

    state = random.randint(0, states - 1)
    total_reward = 0

    for step in range(6):

        probabilities = np.exp(policy[state] - np.max(policy[state]))
        probabilities /= np.sum(probabilities)

        action = np.random.choice(actions, p=probabilities)

        if action == 1:
            reward = 10
        else:
            reward = -1

        # Clipped policy-style update
        update = np.clip(0.05 * reward, -0.2, 0.2)

        policy[state, action] += update

        total_reward += reward
        state = random.randint(0, states - 1)

    if episode % 5 == 0:
        print("Episode:", episode,
              "Reward:", total_reward)

print("\nLearned Walking Policy")

for state in range(states):
    action = np.argmax(policy[state])
    print("State", state, "->", action_names[action])

print("\nPPO/TRPO Training Completed")