import numpy as np
import random

# ============================================================
# EXPERIMENT 31: SARSA FOR TIC-TAC-TOE
# ============================================================

alpha = 0.1
gamma = 0.9
epsilon = 0.2
episodes = 5000

# Q-table
Q = {}


def get_state(board):
    return tuple(board)


def available_moves(board):
    return [i for i in range(9) if board[i] == 0]


def check_winner(board):
    winning_positions = [
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),
        (0, 4, 8),
        (2, 4, 6)
    ]

    for a, b, c in winning_positions:
        if board[a] != 0:
            if board[a] == board[b] == board[c]:
                return board[a]

    if 0 not in board:
        return 0

    return None


def get_q_values(state):
    if state not in Q:
        Q[state] = np.zeros(9)

    return Q[state]


def choose_action(board, explore=True):

    moves = available_moves(board)

    state = get_state(board)
    q_values = get_q_values(state)

    # Exploration
    if explore and random.random() < epsilon:
        return random.choice(moves)

    # Exploitation
    best_value = max(q_values[m] for m in moves)

    best_moves = [
        m for m in moves
        if q_values[m] == best_value
    ]

    return random.choice(best_moves)


# ============================================================
# TRAINING
# ============================================================

for episode in range(episodes):

    board = [0] * 9

    # Initial state and action
    state = get_state(board)
    action = choose_action(board)

    while True:

        # ----------------------------------------------------
        # Agent makes a move
        # ----------------------------------------------------
        board[action] = 1

        result = check_winner(board)

        # Agent wins
        if result == 1:

            reward = 1
            next_state = get_state(board)
            next_action = None

        # Draw
        elif result == 0:

            reward = 0
            next_state = get_state(board)
            next_action = None

        # Game continues
        else:

            # ------------------------------------------------
            # Random opponent move
            # ------------------------------------------------
            opponent_moves = available_moves(board)

            if not opponent_moves:

                reward = 0
                next_state = get_state(board)
                next_action = None

            else:

                opponent_action = random.choice(
                    opponent_moves
                )

                board[opponent_action] = -1

                result = check_winner(board)

                # Opponent wins
                if result == -1:

                    reward = -1
                    next_state = get_state(board)
                    next_action = None

                # Draw
                elif result == 0:

                    reward = 0
                    next_state = get_state(board)
                    next_action = None

                # Continue
                else:

                    reward = 0
                    next_state = get_state(board)

                    next_action = choose_action(
                        board,
                        explore=True
                    )

        # ----------------------------------------------------
        # SARSA UPDATE
        # Q(s,a) = Q(s,a) + alpha[
        # reward + gamma*Q(s',a') - Q(s,a)]
        # ----------------------------------------------------

        current_q = get_q_values(state)[action]

        if next_action is None:

            target = reward

        else:

            next_q = get_q_values(next_state)[next_action]

            target = reward + gamma * next_q

        Q[state][action] += (
            alpha * (target - current_q)
        )

        # ----------------------------------------------------
        # End episode or move to next state
        # ----------------------------------------------------

        if next_action is None:
            break

        state = next_state
        action = next_action


# ============================================================
# EVALUATION
# ============================================================

wins = 0
draws = 0
losses = 0

evaluation_games = 100

for game in range(evaluation_games):

    board = [0] * 9

    while True:

        # Agent move
        moves = available_moves(board)

        if not moves:
            result = 0
            break

        action = choose_action(
            board,
            explore=False
        )

        board[action] = 1

        result = check_winner(board)

        if result is not None:
            break

        # Random opponent
        opponent_moves = available_moves(board)

        if not opponent_moves:
            result = 0
            break

        opponent_action = random.choice(
            opponent_moves
        )

        board[opponent_action] = -1

        result = check_winner(board)

        if result is not None:
            break

    if result == 1:
        wins += 1

    elif result == 0:
        draws += 1

    else:
        losses += 1


# ============================================================
# OUTPUT
# ============================================================

print()
print("=" * 60)
print("SARSA TIC-TAC-TOE AGENT")
print("=" * 60)

print()
print("Training Episodes :", episodes)
print("Evaluation Games  :", evaluation_games)

print()
print("Wins   :", wins)
print("Draws  :", draws)
print("Losses :", losses)

win_rate = (wins / evaluation_games) * 100

print()
print("Win Rate :", round(win_rate, 2), "%")

print()
print("Learned States :", len(Q))

print()
print("Result: SARSA agent trained and evaluated successfully.")

print("=" * 60)