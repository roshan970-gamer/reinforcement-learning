import numpy as np

# Simplified DDPG-style continuous control example.
# Actor chooses a continuous action: amount of resource to spend.
np.random.seed(5)
actor_w = 0.5
critic_w = np.zeros(2)  # [state, action]
lr = 0.01

def actor(state):
    return np.tanh(actor_w * state)

def reward(state, action):
    # Best action is around 0.5 for this toy task.
    return -(action - 0.5)**2 - 0.1*state

for episode in range(500):
    state = np.random.rand()
    action = np.clip(actor(state) + np.random.normal(0,0.1), -1, 1)
    r = reward(state, action)

    # Simple critic update
    x = np.array([state, action])
    pred = critic_w @ x
    critic_w += lr * (r - pred) * x

    # Actor update toward action with better reward
    target_action = 0.5
    actor_w += lr * (target_action - action) * state

print("DDPG-style training completed.")
for s in [0.2, 0.5, 0.8]:
    print("State", s, "Actor action:", round(float(actor(s)), 3))
