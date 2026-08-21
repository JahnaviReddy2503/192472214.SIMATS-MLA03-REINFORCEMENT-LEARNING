import numpy as np
import random
import tensorflow as tf
from collections import deque

# State:
# [lane position, speed, distance, safety distance]

state_size = 4
action_size = 3

# Actions:
# 0 = Slow Down
# 1 = Maintain Speed
# 2 = Accelerate

gamma = 0.95
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.97

learning_rate = 0.001
episodes = 30
batch_size = 16

memory = deque(maxlen=2000)


# Create DQN model
def create_model():

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(state_size,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(action_size, activation="linear")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="mse"
    )

    return model


model = create_model()
target_model = create_model()

target_model.set_weights(model.get_weights())


# Store experience
def remember(state, action, reward, next_state, done):

    memory.append(
        (state, action, reward, next_state, done)
    )


# Train DQN
def replay():

    if len(memory) < batch_size:
        return

    batch = random.sample(memory, batch_size)

    states = []
    targets = []

    for state, action, reward, next_state, done in batch:

        target = model.predict(
            state,
            verbose=0
        )[0]

        if done:

            target[action] = reward

        else:

            future_q = np.max(
                target_model.predict(
                    next_state,
                    verbose=0
                )[0]
            )

            target[action] = (
                reward + gamma * future_q
            )

        states.append(state[0])
        targets.append(target)

    model.fit(
        np.array(states),
        np.array(targets),
        epochs=1,
        verbose=0
    )


print("=" * 60)
print("DQN AUTONOMOUS HIGHWAY VEHICLE")
print("=" * 60)


# Training
for episode in range(episodes):

    state = np.random.rand(
        1,
        state_size
    ).astype(np.float32)

    total_reward = 0

    for step in range(20):

        # Exploration or exploitation
        if random.random() <= epsilon:

            action = random.randrange(action_size)

        else:

            q_values = model.predict(
                state,
                verbose=0
            )

            action = np.argmax(q_values[0])


        # Simulated next state
        next_state = np.random.rand(
            1,
            state_size
        ).astype(np.float32)


        # Safety-based reward
        safety_distance = state[0][3]

        if safety_distance < 0.2:

            reward = -10

        elif action == 2:

            reward = 5

        else:

            reward = 2


        done = (step == 19)

        remember(
            state,
            action,
            reward,
            next_state,
            done
        )

        state = next_state

        total_reward += reward

        replay()

        if done:
            break


    # Reduce exploration
    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )


    # Update target network
    if episode % 5 == 0:

        target_model.set_weights(
            model.get_weights()
        )


    print(
        "Episode:",
        episode + 1,
        "Reward:",
        round(total_reward, 2),
        "Epsilon:",
        round(epsilon, 3)
    )


print("\nTraining completed successfully.")


# Test the trained DQN
test_state = np.array(
    [[0.5, 0.7, 0.6, 0.8]],
    dtype=np.float32
)

q_values = model.predict(
    test_state,
    verbose=0
)[0]


print("\nTest State:")
print("[Lane, Speed, Distance, Safety Distance]")
print(test_state[0])

print("\nPredicted Q Values:")
print(np.round(q_values, 3))


best_action = np.argmax(q_values)

action_names = [
    "Slow Down",
    "Maintain Speed",
    "Accelerate"
]

print("\nRecommended Driving Action:",
      action_names[best_action])

print("\nDQN autonomous driving evaluation completed.")