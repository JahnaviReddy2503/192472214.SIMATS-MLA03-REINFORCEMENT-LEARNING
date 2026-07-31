import gymnasium as gym

# Create FrozenLake environment
env = gym.make("FrozenLake-v1")

state, info = env.reset()

print("Initial State:", state)

for step in range(10):

    action = env.action_space.sample()

    next_state, reward, terminated, truncated, info = env.step(action)

    print("\nStep:", step + 1)
    print("Action:", action)
    print("Next State:", next_state)
    print("Reward:", reward)

    state = next_state

    if terminated or truncated:
        print("\nEpisode Finished")
        break

env.close()
