import numpy as np

# Small Dueling-DQN-style tabular architecture:
# Q(s,a) = V(s) + A(s,a) - mean(A(s,.))
SIZE = 5
goal = (4,4)
V = np.zeros((SIZE,SIZE))
A = np.zeros((SIZE,SIZE,4))
alpha, gamma = 0.1, 0.9

moves = [(-1,0),(1,0),(0,-1),(0,1)]

def step(s,a):
    r,c=s
    dr,dc=moves[a]
    ns=(int(np.clip(r+dr,0,SIZE-1)), int(np.clip(c+dc,0,SIZE-1)))
    reward=10 if ns==goal else -0.1
    return ns,reward

def qvalues(s):
    return V[s] + A[s] - np.mean(A[s])

for ep in range(1000):
    s=(0,0)
    for _ in range(50):
        q=qvalues(s)
        a=np.argmax(q) if np.random.rand()>0.1 else np.random.randint(4)
        ns,r=step(s,a)
        target=r if ns==goal else r+gamma*np.max(qvalues(ns))
        td=target-q[a]
        V[s]+=alpha*td
        A[s][a]+=alpha*td
        s=ns
        if s==goal: break

print("Dueling DQN training completed.")
print("Learned start-state Q values:", np.round(qvalues((0,0)),2))
