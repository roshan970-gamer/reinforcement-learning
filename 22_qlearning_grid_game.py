import numpy as np

SIZE = 5
START, FOOD, GHOST = (0, 0), (4, 4), (2, 2)
ACTIONS = [(-1,0), (1,0), (0,-1), (0,1)]
Q = np.zeros((SIZE, SIZE, 4))
alpha, gamma, epsilon = 0.1, 0.95, 0.2

def step(pos, action):
    r, c = pos
    dr, dc = ACTIONS[action]
    nr, nc = np.clip(r + dr, 0, SIZE-1), np.clip(c + dc, 0, SIZE-1)
    new_pos = (nr, nc)
    if new_pos == FOOD:
        return new_pos, 10, True
    if new_pos == GHOST:
        return new_pos, -10, True
    return new_pos, -0.1, False

for ep in range(1000):
    pos = START
    for _ in range(50):
        action = np.random.randint(4) if np.random.rand() < epsilon else np.argmax(Q[pos])
        new_pos, r, done = step(pos, action)
        Q[pos][action] += alpha * (r + gamma * np.max(Q[new_pos]) * (not done) - Q[pos][action])
        pos = new_pos
        if done:
            break

print("Training complete.")
pos = START
path = [pos]
for _ in range(30):
    action = np.argmax(Q[pos])
    pos, r, done = step(pos, action)
    path.append(pos)
    if done:
        break
print("Learned path:", path)
print("Final reward:", r)
