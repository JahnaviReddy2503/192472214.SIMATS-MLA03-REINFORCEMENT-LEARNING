import random
import numpy as np

# Main Tasks
tasks = ["Clean Room", "Pick Object", "Deliver Object"]

# Subtasks
subtasks = {
    "Clean Room": ["Move", "Vacuum", "Return"],
    "Pick Object": ["Move", "Pick", "Lift"],
    "Deliver Object": ["Move", "Carry", "Drop"]
}

# Q Table
Q = {}

alpha = 0.1
gamma = 0.9

print("="*65)
print("Hierarchical Reinforcement Learning using HAM and MAXQ")
print("="*65)

episodes = 10

for episode in range(episodes):

    print("\nEpisode", episode + 1)

    total_reward = 0

    task = random.choice(tasks)

    print("Main Task :", task)

    for subtask in subtasks[task]:

        state = subtask

        reward = random.randint(5,20)

        if state not in Q:
            Q[state] = 0

        Q[state] = Q[state] + alpha * (reward + gamma * Q[state] - Q[state])

        total_reward += reward

        print(" Subtask :", subtask,
              "| Reward :", reward,
              "| Q Value :", round(Q[state],2))

    print(" Total Reward :", total_reward)

print("\n")
print("="*65)
print("Final Learned Q Values")
print("="*65)

for state in Q:
    print(state,"=",round(Q[state],2))

best_task = max(Q,key=Q.get)

print("\nBest Learned Subtask :",best_task)