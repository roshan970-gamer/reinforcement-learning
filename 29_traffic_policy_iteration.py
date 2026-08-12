import numpy as np

# State: 0=low traffic, 1=high traffic
# Action: 0=short green, 1=long green
states, actions = 2, 2
policy = np.zeros(states, dtype=int)
V = np.zeros(states)
gamma = 0.9

R = np.array([[2, 5], [6, 3]])  # reward for each state/action
P = np.array([
    [[0.7, 0.3], [0.4, 0.6]],
    [[0.2, 0.8], [0.6, 0.4]]
])

for _ in range(20):
    # Policy evaluation
    for _ in range(50):
        newV = np.zeros(states)
        for s in range(states):
            a = policy[s]
            newV[s] = R[s, a] + gamma * np.dot(P[s, a], V)
        V = newV

    # Policy improvement
    stable = True
    for s in range(states):
        values = [R[s,a] + gamma*np.dot(P[s,a], V) for a in range(actions)]
        best = np.argmax(values)
        if best != policy[s]:
            policy[s] = best
            stable = False
    if stable:
        break

print("Optimal state values:", np.round(V, 2))
print("Policy: 0=short green, 1=long green")
print("Best actions:", policy)
