import numpy as np
import random

states = 6
actions = 3

action_names = [
    "Turn Left",
    "Move Forward",
    "Turn Right"
]

policy = np.zeros((states, actions))

learning_rate = 0.05
episodes = 20

print("REINFORCE Autonomous Parking System\n")

for episode in range(1, episodes + 1):

    state = random.randint(0, states - 1)
    episode_data = []
    total_reward = 0

    for step in range(6):

        preferences = policy[state]
        probabilities = np.exp(preferences - np.max(preferences))
        probabilities /= np.sum(probabilities)

        action = np.random.choice(actions, p=probabilities)

        if action == 1:
            reward = 10
        else:
            reward = -2

        episode_data.append((state, action, reward))
        total_reward += reward

        state = random.randint(0, states - 1)

    # REINFORCE update
    G = 0

    for state, action, reward in reversed(episode_data):

        G = reward + 0.9 * G

        probabilities = np.exp(policy[state] - np.max(policy[state]))
        probabilities /= np.sum(probabilities)

        policy[state, action] += learning_rate * G * (
            1 - probabilities[action]
        )

    if episode % 5 == 0:
        print("Episode:", episode,
              "Total Reward:", total_reward)

print("\nLearned Parking Policy")

for state in range(states):
    best_action = np.argmax(policy[state])
    print("State", state, "->", action_names[best_action])

print("\nTraining Completed")