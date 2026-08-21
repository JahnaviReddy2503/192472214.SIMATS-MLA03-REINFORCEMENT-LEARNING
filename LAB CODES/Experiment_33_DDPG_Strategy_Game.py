import numpy as np
import tensorflow as tf

# State and action sizes
STATE_SIZE = 4
ACTION_SIZE = 2

# Actor Network
actor = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(STATE_SIZE,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(ACTION_SIZE, activation="tanh")
])

# Critic Network
critic = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(STATE_SIZE + ACTION_SIZE,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1)
])

actor_optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001
)

critic_optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001
)

gamma = 0.99

print("=" * 60)
print("DDPG STRATEGY GAME AGENT")
print("=" * 60)

# Training
for episode in range(30):

    state = np.random.rand(
        1, STATE_SIZE
    ).astype(np.float32)

    total_reward = 0

    for step in range(20):

        # Actor selects action
        action = actor(state)

        # Exploration noise
        noise = tf.random.normal(
            shape=action.shape,
            mean=0.0,
            stddev=0.1
        )

        action_with_noise = tf.clip_by_value(
            action + noise,
            -1.0,
            1.0
        )

        # Simulated strategy game
        resource_action = tf.maximum(
            0.0,
            action_with_noise[0, 0]
        )

        unit_action = tf.maximum(
            0.0,
            action_with_noise[0, 1]
        )

        reward = (
            5.0 * resource_action
            + 3.0 * unit_action
        )

        total_reward += float(reward)

        # Next state
        next_state = np.random.rand(
            1, STATE_SIZE
        ).astype(np.float32)

        # Target action
        next_action = actor(next_state)

        next_critic_input = tf.concat(
            [
                tf.convert_to_tensor(next_state),
                next_action
            ],
            axis=1
        )

        next_value = critic(next_critic_input)

        target = reward + gamma * next_value

        # Train Critic
        with tf.GradientTape() as tape:

            critic_input = tf.concat(
                [
                    tf.convert_to_tensor(state),
                    action_with_noise
                ],
                axis=1
            )

            predicted_value = critic(
                critic_input
            )

            critic_loss = tf.reduce_mean(
                tf.square(
                    target - predicted_value
                )
            )

        critic_gradients = tape.gradient(
            critic_loss,
            critic.trainable_variables
        )

        critic_optimizer.apply_gradients(
            zip(
                critic_gradients,
                critic.trainable_variables
            )
        )

        # Train Actor
        with tf.GradientTape() as tape:

            predicted_action = actor(state)

            actor_input = tf.concat(
                [
                    tf.convert_to_tensor(state),
                    predicted_action
                ],
                axis=1
            )

            actor_loss = -tf.reduce_mean(
                critic(actor_input)
            )

        actor_gradients = tape.gradient(
            actor_loss,
            actor.trainable_variables
        )

        actor_optimizer.apply_gradients(
            zip(
                actor_gradients,
                actor.trainable_variables
            )
        )

        state = next_state

    if (episode + 1) % 5 == 0:
        print(
            "Episode:",
            episode + 1,
            "Total Reward:",
            round(total_reward, 2)
        )

# Testing
test_state = np.array(
    [[0.5, 0.6, 0.4, 0.8]],
    dtype=np.float32
)

test_action = actor(
    test_state,
    training=False
).numpy()[0]

print("\n" + "=" * 60)
print("TESTING")
print("=" * 60)

print("Test Game State:",
      test_state[0])

print(
    "DDPG Action:",
    np.round(test_action, 3)
)

print(
    "\nResource Gathering Action:",
    round(float(test_action[0]), 3)
)

print(
    "Unit Building Action:",
    round(float(test_action[1]), 3)
)

print(
    "\nResult: DDPG strategy game agent "
    "trained successfully."
)