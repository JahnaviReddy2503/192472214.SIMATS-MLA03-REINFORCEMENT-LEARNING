import numpy as np
import random

prices = [50, 60, 70, 80, 90]

# Probability of purchase for each price
purchase_probability = [0.80, 0.70, 0.60, 0.45, 0.30]

trials = 500


def epsilon_greedy(epsilon=0.1):

    counts = np.zeros(len(prices))
    revenue = np.zeros(len(prices))

    for _ in range(trials):

        if random.random() < epsilon:
            index = random.randrange(len(prices))
        else:
            index = np.argmax(
                revenue / (counts + 1e-9)
            )

        sale = random.random() < purchase_probability[index]

        value = prices[index] if sale else 0

        counts[index] += 1
        revenue[index] += value

    return revenue.sum()


def ucb():

    counts = np.ones(len(prices))
    revenue = np.zeros(len(prices))

    for t in range(len(prices), trials):

        average = revenue / counts

        confidence = np.sqrt(
            2 * np.log(t + 1) / counts
        )

        index = np.argmax(average + confidence)

        sale = random.random() < purchase_probability[index]

        value = prices[index] if sale else 0

        counts[index] += 1
        revenue[index] += value

    return revenue.sum()


def thompson():

    success = np.ones(len(prices))
    failure = np.ones(len(prices))

    total_revenue = 0

    for _ in range(trials):

        samples = np.random.beta(
            success,
            failure
        )

        index = np.argmax(
            samples * prices
        )

        sale = random.random() < purchase_probability[index]

        if sale:
            success[index] += 1
            total_revenue += prices[index]
        else:
            failure[index] += 1

    return total_revenue


results = {
    "Epsilon-Greedy": epsilon_greedy(),
    "UCB": ucb(),
    "Thompson Sampling": thompson()
}

print("=" * 65)
print("DYNAMIC PRICING USING BANDIT ALGORITHMS")
print("=" * 65)

for name, revenue in results.items():
    print(name, "Revenue =", round(revenue, 2))

best = max(results, key=results.get)

print("\nBest Pricing Strategy:", best)
print("Maximum Revenue:", round(results[best], 2))