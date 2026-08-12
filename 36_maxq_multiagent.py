import numpy as np

# Toy hierarchical task:
# ROOT -> collect resource -> build unit -> finish
# Two agents cooperate by sharing one task state.
Q_collect = np.zeros((3,2))
Q_build = np.zeros((3,2))
alpha, gamma = 0.1, 0.9

for episode in range(500):
    state = 0
    # MAXQ top-level decomposition
    for _ in range(10):
        # Subtask 1: collect
        action = np.argmax(Q_collect[state]) if np.random.rand()>0.2 else np.random.randint(2)
        next_state = min(2, state+1) if action == 1 else state
        reward = 2 if action == 1 else -1
        Q_collect[state,action] += alpha*(reward + gamma*np.max(Q_collect[next_state])-Q_collect[state,action])
        state = next_state

        # Subtask 2: build when enough resources
        if state == 2:
            action = np.argmax(Q_build[state]) if np.random.rand()>0.2 else np.random.randint(2)
            reward = 10 if action == 1 else -2
            Q_build[state,action] += alpha*(reward-Q_build[state,action])
            break

print("MAXQ-style hierarchical training completed.")
print("Collect policy:", np.argmax(Q_collect,axis=1))
print("Build policy:", np.argmax(Q_build,axis=1))
print("Agent cooperation objective: collect resources -> build -> finish")
