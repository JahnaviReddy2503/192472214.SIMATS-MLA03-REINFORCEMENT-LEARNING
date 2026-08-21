import numpy as np

np.random.seed(7)

STATES = 3
GOAL = 2
EPISODES = 100

def observe(actual):
    if np.random.rand() < 0.80:
        return actual
    return np.random.randint(STATES)

def update_belief(belief,observation):
    likelihood = np.full(STATES,0.10)
    likelihood[observation] = 0.80
    posterior = belief*likelihood
    return posterior/posterior.sum()

success = 0

for _ in range(EPISODES):
    actual = np.random.randint(STATES)
    belief = np.ones(STATES)/STATES

    for _ in range(10):
        observation = observe(actual)
        belief = update_belief(belief,observation)
        estimate = int(np.argmax(belief))

        if estimate < GOAL:
            actual = min(GOAL,actual+1)
        else:
            actual = GOAL

        if actual == GOAL:
            success += 1
            break

print("="*65)
print("EXPERIMENT 38 - POMDP ROBOT NAVIGATION")
print("="*65)
print("Sensor accuracy: 80%")
print("Episodes:",EPISODES)
print("Successful navigation:",success)
print("Success rate:",round(100*success/EPISODES,2),"%")
print("Final belief state:",np.round(belief,3))
print("Result: POMDP navigation completed successfully.")
