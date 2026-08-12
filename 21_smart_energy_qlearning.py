import numpy as np

# States: 0=low battery, 1=medium, 2=high
# Actions: 0=use grid, 1=use solar, 2=use battery
Q = np.zeros((3, 3))
alpha, gamma, epsilon = 0.1, 0.9, 0.2

def reward(state, action):
    # Simple safe/fair energy rule
    table = [
        [-2,  3,  1],   # low battery
        [-1,  4,  3],   # medium
        [-1,  5,  2]    # high
    ]
    return table[state][action]

for episode in range(500):
    state = np.random.randint(3)
    for _ in range(20):
        if np.random.rand() < epsilon:
            action = np.random.randint(3)
        else:
            action = np.argmax(Q[state])

        r = reward(state, action)
        next_state = np.clip(state + np.random.choice([-1, 0, 1]), 0, 2)
        Q[state, action] += alpha * (
            r + gamma * np.max(Q[next_state]) - Q[state, action]
        )
        state = next_state

print("Q-table:")
print(np.round(Q, 2))
print("Best action for each state:", np.argmax(Q, axis=1))
print("0=Grid, 1=Solar, 2=Battery")
