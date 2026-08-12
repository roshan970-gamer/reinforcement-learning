import numpy as np

# State = student mastery: 0=low, 1=medium, 2=high
# Actions = 0=easy lesson, 1=normal lesson, 2=hard lesson
Q=np.zeros((3,3))
alpha,gamma,epsilon=0.1,0.9,0.2

for ep in range(1000):
    state=np.random.randint(3)
    for _ in range(15):
        action=np.random.randint(3) if np.random.rand()<epsilon else np.argmax(Q[state])

        if state==0:
            reward={0:3,1:1,2:-2}[action]
        elif state==1:
            reward={0:1,1:3,2:2}[action]
        else:
            reward={0:0,1:2,2:4}[action]

        next_state=np.clip(state + (1 if reward>=2 else 0) + np.random.choice([-1,0,1]),0,2)
        Q[state,action]+=alpha*(reward+gamma*np.max(Q[next_state])-Q[state,action])
        state=next_state

print("Personalized education RL training completed.")
print("Best lesson for mastery 0/1/2:",np.argmax(Q,axis=1))
print("0=easy, 1=normal, 2=hard")
