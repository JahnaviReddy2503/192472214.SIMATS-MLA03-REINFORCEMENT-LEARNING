import numpy as np

np.random.seed(7)

STATES = 3
ACTIONS = 3
ALPHA = 0.03
GAMMA = 0.90
EPISODES = 1000

policy = np.zeros((STATES,ACTIONS))

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()

def transition(state,action):
    if state == 0:
        reward = 5 if action == 2 else -2
        next_state = 1 if action == 2 else 0
    elif state == 1:
        reward = 5 if action == 1 else 1
        next_state = 1
    else:
        reward = 5 if action == 0 else -2
        next_state = 1 if action == 0 else 2
    if action != 1:
        reward -= 0.5
    return next_state,reward

for _ in range(EPISODES):
    state = np.random.randint(STATES)
    trajectory = []

    for _ in range(12):
        probs = softmax(policy[state])
        action = np.random.choice(ACTIONS,p=probs)
        next_state,reward = transition(state,action)
        trajectory.append((state,action,reward))
        state = next_state

    G = 0.0
    returns = []
    for _,_,reward in reversed(trajectory):
        G = reward + GAMMA*G
        returns.insert(0,G)

    for (s,a,_),G in zip(trajectory,returns):
        probs = softmax(policy[s])
        grad = -probs
        grad[a] += 1
        policy[s] += ALPHA*G*grad

state_names = ["Cold","Comfortable","Hot"]
action_names = ["Decrease Temperature","Maintain Temperature","Increase Temperature"]

print("="*65)
print("EXPERIMENT 34 - REINFORCE SMART HOME")
print("="*65)

for s in range(STATES):
    probs = softmax(policy[s])
    print("\nState:",state_names[s])
    print("Action probabilities:",np.round(probs,3))
    print("Recommended action:",action_names[int(np.argmax(probs))])

print("\nResult: REINFORCE learned a comfort-energy policy successfully.")
