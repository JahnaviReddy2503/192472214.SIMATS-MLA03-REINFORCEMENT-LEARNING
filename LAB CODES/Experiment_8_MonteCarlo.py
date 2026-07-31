import random

episodes = 10

returns = []

print("Monte Carlo Prediction\n")

for episode in range(1, episodes + 1):

    reward = random.randint(5,15)

    returns.append(reward)

    average = sum(returns)/len(returns)

    print("Episode:", episode)
    print("Reward:", reward)
    print("Average Return:", round(average,2))
    print()

print("Learning Completed")
