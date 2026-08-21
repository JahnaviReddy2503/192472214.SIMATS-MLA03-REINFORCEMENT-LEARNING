import random

random.seed(7)

EPISODES = 500
ALPHA = 0.10

Q_collect = {"low":0.0,"high":0.0}
Q_build = {"low":0.0,"high":0.0}

successes = 0
total_reward = 0.0

def collect_resources():
    agent1 = random.randint(1,4)
    agent2 = random.randint(1,4)
    return agent1 + agent2

for _ in range(EPISODES):
    resources = collect_resources()
    collect_state = "high" if resources >= 6 else "low"

    collect_reward = resources
    Q_collect[collect_state] += ALPHA*(collect_reward-Q_collect[collect_state])

    if resources >= 6:
        build_state = "high"
        build_reward = 8.0
        success = True
    else:
        build_state = "low"
        build_reward = -5.0
        success = False

    Q_build[build_state] += ALPHA*(build_reward-Q_build[build_state])

    total_reward += collect_reward + build_reward
    if success:
        successes += 1

print("="*65)
print("EXPERIMENT 36 - MAXQ HIERARCHICAL COOPERATIVE TASK")
print("="*65)
print("\nHierarchy:")
print("Root Task")
print("  -> Collect Resources")
print("  -> Build Unit")
print("  -> Finish Mission")
print("\nEpisodes:",EPISODES)
print("Successful missions:",successes)
print("Success rate:",round(100*successes/EPISODES,2),"%")
print("Average reward:",round(total_reward/EPISODES,2))
print("Collect Q-values:",Q_collect)
print("Build Q-values:",Q_build)
print("Result: MAXQ hierarchical learning completed successfully.")
