from collections import deque
from copy import deepcopy
import warnings # pylint: disable=unused-import
import numpy as np


class ReplayBuffer():

  def __init__(self, max_size=0) -> None:
    self.max_size = max_size
    self.q = deque([], maxlen=max_size)

  def sample_from(self,
                  num_samples=1,
                  drop_samples=False,
                  sample_distribution_fn=None):
    selected_sample_ids = []
    selected_sample_ids = np.random.choice(
        range(len(self.q)),
        size=(num_samples,),
        replace=True,
        p=sample_distribution_fn() if sample_distribution_fn else None)

    samples = [deepcopy(self.q[idx]) for idx in selected_sample_ids]

    if drop_samples:
      for idx in selected_sample_ids:
        del self.q[idx]

    return samples

  def enqueue(self, sample):
    if not self.isfull():
      return self.q.append(sample)
    # warnings.warn("the buffer is full, the first sample will be dropped.")
    self.q.popleft()
    return self.q.append(sample)

  def dequeue(self):
    return self.q.popleft()

  def __len__(self):
    return len(self.q)

  def isempty(self):
    return len(self) == 0

  def isfull(self):
    if self.max_size:
      return len(self) >= self.max_size
    return False

  def __repr__(self) -> str:
    return repr(self.q)

  def __str__(self) -> str:
    return str(self.q)
