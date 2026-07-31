# Experiment 5 - Epsilon Greedy

import random

rewards = [1, 2, 3]

epsilon = 0.1

total = 0

for i in range(20):

    if random.random() < epsilon:
        action = random.randint(0, 2)
    else:
        action = rewards.index(max(rewards))

    total += rewards[action]

print("Total Reward =", total)
