import numpy as np
import tensorflow as tf

SIZE = 5
STATE_SIZE = 2
ACTION_SIZE = 4

GOAL = np.array([4, 4], dtype=np.float32)


# Move the agent
def move(state, action):

    row = int(state[0])
    col = int(state[1])

    if action == 0:       # Up
        row = max(0, row - 1)

    elif action == 1:     # Down
        row = min(SIZE - 1, row + 1)

    elif action == 2:     # Left
        col = max(0, col - 1)

    elif action == 3:     # Right
        col = min(SIZE - 1, col + 1)

    return np.array([row, col], dtype=np.float32)


# Standard DQN
def create_dqn():

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(STATE_SIZE,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(ACTION_SIZE)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="mse"
    )

    return model


# Dueling DQN
def create_dueling_dqn():

    inputs = tf.keras.Input(
        shape=(STATE_SIZE,)
    )

    x = tf.keras.layers.Dense(
        32,
        activation="relu"
    )(inputs)

    x = tf.keras.layers.Dense(
        32,
        activation="relu"
    )(x)

    # State value
    value = tf.keras.layers.Dense(
        1
    )(x)

    # Action advantages
    advantage = tf.keras.layers.Dense(
        ACTION_SIZE
    )(x)

    # Remove mean advantage
    advantage_mean = tf.keras.layers.Lambda(
        lambda x: x - tf.reduce_mean(
            x,
            axis=1,
            keepdims=True
        )
    )(advantage)

    # Q(s,a) = V(s) + A(s,a)
    outputs = tf.keras.layers.Add()([
        value,
        advantage_mean
    ])

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="mse"
    )

    return model


# Create models
standard_dqn = create_dqn()
dueling_dqn = create_dueling_dqn()


print("=" * 60)
print("EXPERIMENT 32: DUELING DQN VS STANDARD DQN")
print("=" * 60)

print("\nGrid Size:", SIZE, "x", SIZE)
print("Start State: (0, 0)")
print("Goal State: (4, 4)")
print("Actions: 0=Up, 1=Down, 2=Left, 3=Right")

print("\nTraining Dueling DQN...")

# Training
episodes = 100

for episode in range(episodes):

    state = np.array(
        [0, 0],
        dtype=np.float32
    )

    for step in range(30):

        state_input = state.reshape(1, 2)

        q_values = dueling_dqn.predict(
            state_input,
            verbose=0
        )[0]

        action = int(np.argmax(q_values))

        next_state = move(
            state,
            action
        )

        # Reward
        if np.array_equal(
            next_state,
            GOAL
        ):
            reward = 10
        else:
            distance = np.linalg.norm(
                GOAL - next_state
            )
            reward = -0.1 * distance

        # Target Q-values
        target_q = dueling_dqn.predict(
            next_state.reshape(1, 2),
            verbose=0
        )[0]

        target_q[action] = (
            reward +
            0.9 * np.max(target_q)
        )

        dueling_dqn.fit(
            state_input,
            target_q.reshape(1, 4),
            epochs=1,
            verbose=0
        )

        state = next_state

        if np.array_equal(
            state,
            GOAL
        ):
            break


# Compare both models
test_state = np.array(
    [[0, 0]],
    dtype=np.float32
)

standard_q = standard_dqn.predict(
    test_state,
    verbose=0
)[0]

dueling_q = dueling_dqn.predict(
    test_state,
    verbose=0
)[0]


print("\n" + "-" * 60)
print("STANDARD DQN Q-VALUES")
print("-" * 60)

print(
    np.round(
        standard_q,
        3
    )
)


print("\n" + "-" * 60)
print("DUELING DQN Q-VALUES")
print("-" * 60)

print(
    np.round(
        dueling_q,
        3
    )
)


print("\nBest Standard DQN Action:",
      int(np.argmax(standard_q)))

print("Best Dueling DQN Action:",
      int(np.argmax(dueling_q)))


print("\nAction Meaning:")
print("0 = Up")
print("1 = Down")
print("2 = Left")
print("3 = Right")

print("\nRESULT:")
print("Dueling DQN training and comparison completed successfully.")