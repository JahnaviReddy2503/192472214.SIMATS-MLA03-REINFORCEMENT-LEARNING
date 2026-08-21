import numpy as np
import random

ads = 5
trials = 500

true_ctr = [0.10, 0.18, 0.12, 0.25, 0.15]


def epsilon_greedy(epsilon=0.1):

    counts = np.zeros(ads)
    rewards = np.zeros(ads)

    for _ in range(trials):

        if random.random() < epsilon:
            ad = random.randrange(ads)
        else:
            ad = np.argmax(rewards / (counts + 1e-9))

        reward = 1 if random.random() < true_ctr[ad] else 0

        counts[ad] += 1
        rewards[ad] += reward

    return rewards.sum(), rewards / (counts + 1e-9)


def ucb():

    counts = np.ones(ads)
    rewards = np.zeros(ads)

    for t in range(ads, trials):

        values = rewards / counts
        confidence = np.sqrt(
            2 * np.log(t + 1) / counts
        )

        ad = np.argmax(values + confidence)

        reward = 1 if random.random() < true_ctr[ad] else 0

        counts[ad] += 1
        rewards[ad] += reward

    return rewards.sum(), rewards / counts


def thompson_sampling():

    successes = np.ones(ads)
    failures = np.ones(ads)

    total = 0

    for _ in range(trials):

        samples = np.random.beta(
            successes,
            failures
        )

        ad = np.argmax(samples)

        reward = 1 if random.random() < true_ctr[ad] else 0

        if reward:
            successes[ad] += 1
            total += 1
        else:
            failures[ad] += 1

    return total


eg_reward, eg_ctr = epsilon_greedy()
ucb_reward, ucb_ctr = ucb()
ts_reward = thompson_sampling()

print("=" * 65)
print("ADVERTISEMENT BANDIT COMPARISON")
print("=" * 65)

print("\nEpsilon-Greedy Clicks:", int(eg_reward))
print("Estimated CTR:", np.round(eg_ctr, 3))

print("\nUCB Clicks:", int(ucb_reward))
print("Estimated CTR:", np.round(ucb_ctr, 3))

print("\nThompson Sampling Clicks:", int(ts_reward))

rewards = {
    "Epsilon-Greedy": eg_reward,
    "UCB": ucb_reward,
    "Thompson Sampling": ts_reward
}

best = max(rewards, key=rewards.get)

print("\nBest Algorithm:", best)
print("Highest Clicks:", int(rewards[best]))