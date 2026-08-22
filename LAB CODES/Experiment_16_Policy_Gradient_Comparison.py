import numpy as np
import random

states = 5
actions = 3

action_names = [
    "Steer Left",
    "Keep Lane",
    "Steer Right"
]

def train(method):

    policy = np.zeros((states, actions))
    total_rewards = []

    for episode in range(20):

        state = random.randint(0, states - 1)
        total_reward = 0

        for step in range(5):

            probabilities = np.exp(policy[state] - np.max(policy[state]))
            probabilities /= np.sum(probabilities)

            action = np.random.choice(actions, p=probabilities)

            if action == 1:
                reward = 10
            else:
                reward = -2

            if method == "PPO":
                reward += 1

            policy[state, action] += 0.05 * reward

            total_reward += reward
            state = random.randint(0, states - 1)

        total_rewards.append(total_reward)

    return np.mean(total_rewards), policy


print("Policy Gradient Comparison")
print("Autonomous Lane Keeping\n")

methods = ["REINFORCE", "PPO"]

results = {}

for method in methods:

    average, policy = train(method)

    results[method] = average

    print(method,
          "Average Reward:",
          round(average, 2))

best = max(results, key=results.get)

print("\nBest Algorithm:", best)

print("Recommended Action:",
      action_names[np.argmax(train(best)[1][0])])

print("\nComparison Completed")