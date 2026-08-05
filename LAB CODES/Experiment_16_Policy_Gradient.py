# Experiment 16
# Policy Gradient Algorithms for Autonomous Lane Keeping

algorithms = ["REINFORCE", "PPO", "TRPO"]

accuracy = [85, 94, 91]

print("Policy Gradient Comparison\n")

for i in range(len(algorithms)):
    print(algorithms[i], "Driving Accuracy =", accuracy[i], "%")

best = accuracy.index(max(accuracy))

print("\nBest Algorithm :", algorithms[best])
print("Maximum Accuracy :", accuracy[best], "%")
