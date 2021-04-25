import numpy as np

def discount_rewards(rewards):
    discounted_rewards = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(0, len(rewards))):
        running_add = running_add * 0.99 + rewards[t]
        discounted_rewards[t] = running_add
    return discounted_rewards

print(discount_rewards([0,0,0,0,0,0,0,1]))