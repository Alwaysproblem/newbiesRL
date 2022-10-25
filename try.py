# %%
import numpy as np
from collections import deque
import torch 
import torch.nn as nn
# %%
# a = [1,2,3,4,5,6,7]
# a = [1,1,2,2,3,4,5,6,7]
# %%
# np.random.choice( a , size = (7), replace = False, p = [0.5,  0.5])

# %%
class Q(nn.Module):

  def __init__(self, state_dim, action_space, hidden_size=None):
    super().__init__()
    self.action_space = action_space
    self.hidden_size = ((5, 5, 5) if not hidden_size else hidden_size)

    self.hidden_layers = [
        (nn.Linear(in_size, out_size), nn.GELU())
        for in_size, out_size in zip((state_dim,) +
                                     self.hidden_size, self.hidden_size +
                                     (action_space,))
    ]

  def forward(self, state):
    x = state
    for l, a in self.hidden_layers:
      x = a(l(x))
    return x

# %%
q = Q(4, 2)
# %%
a = q(torch.Tensor([1.,2.,3.,4.]))
# %%
a.argmax()
# %%
a
# %%
