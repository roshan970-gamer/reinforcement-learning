import numpy as np
from collections import defaultdict

Q = defaultdict(lambda: np.zeros(9))
alpha, gamma, epsilon = 0.2, 0.95, 0.2

def available(board):
    return [i for i, x in enumerate(board) if x == 0]

def choose(board):
    moves = available(board)
    if np.random.rand() < epsilon:
        return np.random.choice(moves)
    return max(moves, key=lambda m: Q[tuple(board)][m])

def winner(b):
    lines = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
    for a,c,d in lines:
        if b[a] != 0 and b[a] == b[c] == b[d]:
            return b[a]
    return 0

for ep in range(5000):
    board = [0]*9
    state = tuple(board)
    action = choose(board)
    board[action] = 1

    for _ in range(5):
        # Opponent
        moves = available(board)
        if not moves: break
        board[np.random.choice(moves)] = -1
        if winner(board) == -1: reward = -10; break

        if not available(board): reward = 0; break
        next_action = choose(board)
        board[next_action] = 1

        w = winner(board)
        reward = 10 if w == 1 else 0
        next_state = tuple(board)
        Q[state][action] += alpha * (reward + gamma*Q[next_state][next_action] - Q[state][action])
        state, action = next_state, next_action
        if w == 1 or not available(board): break

print("SARSA training completed.")
print("Number of learned states:", len(Q))
