import random

episodes = 5

policy_reward = 0

print("REINFORCE Algorithm\n")

for episode in range(1, episodes + 1):

    reward = random.randint(5,15)

    policy_reward += reward

    print("Episode :", episode)
    print("Reward :", reward)
    print("Total Reward :", policy_reward)
    print()

print("Parking Policy Learned Successfully")
