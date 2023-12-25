"""The buffer protocol described here"""
from collections import deque
from copy import deepcopy
import warnings
import numpy as np


class Experience:
  # pylint: disable=line-too-long
  """Experience is a pickle of (state, action, reward, next_state, done, log_prob...)"""

  def __init__(
      self,
      state=None,
      action=None,
      reward=None,
      next_state=None,
      done=None,
      log_prob=None,
      **kwargs
  ) -> None:
    self.state = state
    self.action = action
    self.reward = reward
    self.next_state = next_state
    self.done = done
    self.log_prob = log_prob
    if kwargs:
      for k, v in kwargs.items():
        setattr(self, k, v)


class Trajectory:
  # pylint: disable=line-too-long
  """
  The Trajectory class is used to store the experiences of a whole trajectory policy.

  The trajectory is a list of experiences, where each experience is a tuple
  of (state, action, reward, next_state, done).

  The trajectory is a list of experiences, where each experience is a tuple
  """

  def __init__(self):
    self.q = deque([])

  def enqueue(self, experience):
    self.q.append(experience)

  def dequeue(self):
    pass

  def is_empty(self):
    return len(self.q) == 0

  def __contain__(self, e):
    return e in self.q

  def __len__(self):
    return len(self.q)

  def __getitem__(self, index):
    return self.q[index]

  def __iter__(self):
    return iter(self.q)

  def __repr__(self) -> str:
    return f"{self.__class__.__name__}({', '.join(repr(e) for e in self.q)})"

  def __str__(self) -> str:
    return f"{self.__class__.__name__}({', '.join(str(e) for e in self.q)})"


class ReplayBuffer():
  """Replay Buffer for DQN training"""

  def __init__(self, max_size=None) -> None:
    self.max_size = max_size
    self.q: deque = deque([], maxlen=max_size)

  def sample_from(
      self,
      sample_ratio=None,
      num_samples=1,
      drop_samples=False,
      sample_distribution_fn=None,
      replace=True,
  ):
    """Sample a batch of experiences from the replay buffer"""
    if not self.q:
      raise ValueError("please use `enqueue` before `sample_from`")

    if sample_ratio:
      num_samples = max(int(len(self.q) * sample_ratio), 1)

    if len(self.q) < num_samples:
      return []

    selected_sample_ids = np.random.choice(
        range(len(self.q)),
        size=(num_samples, ),
        replace=replace,
        p=sample_distribution_fn() if sample_distribution_fn else None
    )

    samples = [deepcopy(self.q[idx]) for idx in selected_sample_ids]

    if drop_samples:
      for idx in selected_sample_ids:
        self.q[idx] = None
      self.delete_none()

    return samples

  def enqueue(self, sample):
    if not self.isfull():
      return self.q.append(sample)
    warnings.warn("the buffer is full, the first sample will be dropped.")
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

  def delete_none(self):
    self.q = deque([sample for sample in self.q if sample is not None])

  def __repr__(self) -> str:
    return f"{self.__class__.__name__}({', '.join(repr(e) for e in self.q)})"

  def __str__(self) -> str:
    return f"{self.__class__.__name__}({', '.join(str(e) for e in self.q)})"

  def __contain__(self, e):
    return e in self.q

  def __getitem__(self, index):
    return self.q[index]

  def __iter__(self):
    return iter(self.q)
