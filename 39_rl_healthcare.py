import numpy as np

# State = patient queue level: 0=low, 1=medium, 2=high
# Action = 0=normal resources, 1=extra resources
Q=np.zeros((3,2))
alpha,gamma,epsilon=0.1,0.9,0.2

for ep in range(1000):
    state=np.random.randint(3)
    for _ in range(20):
        action=np.random.randint(2) if np.random.rand()<epsilon else np.argmax(Q[state])
        wait_cost= -2*state
        resource_cost= -1 if action==1 else 0
        outcome=3 if action==1 and state==2 else 1
        reward=wait_cost+resource_cost+outcome

        next_state=np.clip(state+np.random.choice([-1,0,1]),0,2)
        Q[state,action]+=alpha*(reward+gamma*np.max(Q[next_state])-Q[state,action])
        state=next_state

print("Healthcare RL training completed.")
print("Best action per queue state:",np.argmax(Q,axis=1))
print("0=normal resources, 1=extra resources")