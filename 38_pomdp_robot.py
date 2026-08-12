import numpy as np

# Tiny POMDP: robot does not directly know its exact location.
# It maintains a belief: probability of being in each cell.
states = 4
belief = np.array([1.0,0,0,0])
transition = np.array([
    [0.8,0.2,0,0],
    [0.1,0.8,0.1,0],
    [0,0.1,0.8,0.1],
    [0,0,0.2,0.8]
])
goal = 3

for step in range(10):
    # Move belief
    belief = belief @ transition

    # Simple noisy sensor: sensor says "goal nearby" with probability
    # depending on belief in the final state.
    observation_prob = 0.8*belief[goal] + 0.2*(1-belief[goal])
    observed_goal = np.random.rand() < observation_prob

    # Belief update
    likelihood = np.array([0.2,0.2,0.2,0.8]) if observed_goal else np.array([0.8,0.8,0.8,0.2])
    belief = belief*likelihood
    belief /= belief.sum()

    print(f"Step {step+1}: belief =", np.round(belief,3))

print("Most likely robot location:", np.argmax(belief))
