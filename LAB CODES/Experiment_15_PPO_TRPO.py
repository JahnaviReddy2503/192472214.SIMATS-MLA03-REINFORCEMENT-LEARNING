print("Humanoid Robot Walking\n")

algorithms = ["PPO","TRPO"]

balance_score = [92,89]

for i in range(len(algorithms)):

    print(algorithms[i], "Balance Score =", balance_score[i])

best = balance_score.index(max(balance_score))

print("\nBest Algorithm :", algorithms[best])
print("Highest Stability :", balance_score[best])
