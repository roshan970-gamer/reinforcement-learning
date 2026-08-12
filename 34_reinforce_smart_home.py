import numpy as np

# State = room temperature category: 0=cold, 1=comfortable, 2=hot
# Action = 0=decrease heating, 1=keep, 2=increase heating
policy = np.zeros((3,3))
alpha, gamma = 0.05, 0.9

def softmax(x):
    e=np.exp(x-np.max(x))
    return e/e.sum()

for ep in range(1000):
    state=np.random.randint(3)
    states=[]; actions=[]; rewards=[]

    for _ in range(15):
        p=softmax(policy[state])
        action=np.random.choice(3,p=p)
        comfort = 5 if (state==1 and action==1) else 2
        energy = -2 if action != 1 else 0
        r=comfort+energy
        states.append(state); actions.append(action); rewards.append(r)
        state=np.random.randint(3)

    G=0
    for t in reversed(range(len(rewards))):
        G=rewards[t]+gamma*G
        p=softmax(policy[states[t]])
        grad=-p
        grad[actions[t]]+=1
        policy[states[t]]+=alpha*G*grad

print("REINFORCE smart-home training completed.")
print("Best action by temperature:", np.argmax(policy,axis=1))
print("0=decrease, 1=keep, 2=increase")
