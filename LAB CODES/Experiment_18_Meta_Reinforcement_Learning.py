import numpy as np
import random

tasks = {
    "Task 1": 5,
    "Task 2": 8,
    "Task 3": 10
}

actions = 3

action_names = [
    "Slow",
    "Normal",
    "Fast"
]

meta_policy = np.zeros(actions)

episodes = 20

print("Meta-Reinforcement Learning")
print("Adaptive Industrial Robot\n")

for episode in range(1, episodes + 1):

    task_name = random.choice(list(tasks.keys()))
    target = tasks[task_name]

    state = random.randint(1, 10)

    if state < target:
        best_action = 2
    elif state > target:
        best_action = 0
    else:
        best_action = 1

    # Meta-policy learns useful action
    meta_policy[best_action] += 0.1

    reward = 10 if best_action == 1 else 5

    if episode % 5 == 0:
        print("Episode:", episode)
        print("Task:", task_name)
        print("Reward:", reward)
        print()

print("Learned Meta Policy")

best_action = np.argmax(meta_policy)

print("Best Adaptive Action:",
      action_names[best_action])

print("\nMeta-Reinforcement Learning Completed")