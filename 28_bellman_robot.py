import numpy as np

SIZE = 4
goal = (3, 3)
obstacles = {(1, 1)}
gamma = 0.9
V = np.zeros((SIZE, SIZE))
actions = [(-1,0), (1,0), (0,-1), (0,1)]

for _ in range(100):
    old = V.copy()
    for r in range(SIZE):
        for c in range(SIZE):
            if (r, c) == goal or (r, c) in obstacles:
                continue
            values = []
            for dr, dc in actions:
                nr, nc = np.clip(r+dr, 0, SIZE-1), np.clip(c+dc, 0, SIZE-1)
                if (nr, nc) in obstacles:
                    nr, nc = r, c
                reward = 10 if (nr, nc) == goal else -1
                values.append(reward + gamma * old[nr, nc])
            V[r, c] = max(values)

print("Optimal state-value function:")
print(np.round(V, 2))

pos = (0, 0)
path = [pos]
while pos != goal and len(path) < 20:
    r, c = pos
    best = None
    best_val = -1e9
    for dr, dc in actions:
        nr, nc = np.clip(r+dr, 0, SIZE-1), np.clip(c+dc, 0, SIZE-1)
        if (nr, nc) in obstacles:
            continue
        if V[nr, nc] > best_val:
            best_val = V[nr, nc]
            best = (nr, nc)
    pos = best
    path.append(pos)

print("Optimal path:", path)
