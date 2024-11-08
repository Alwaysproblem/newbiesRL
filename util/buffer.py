"""The buffer protocol described here"""
from collections import deque
from copy import deepcopy
import warnings
import math
import random
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
    return f"{self.__class__.__name__}({", ".join(repr(e) for e in self.q)})"

  def __str__(self) -> str:
    return f"{self.__class__.__name__}({", ".join(str(e) for e in self.q)})"


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
    self._dequeue()
    return self.q.append(sample)

  def dequeue(self):
    return self._dequeue()

  def _dequeue(self):
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
    return f"{self.__class__.__name__}({", ".join(repr(e) for e in self.q)})"

  def __str__(self) -> str:
    return f"{self.__class__.__name__}({", ".join(str(e) for e in self.q)})"

  def __contain__(self, e):
    return e in self.q

  def __getitem__(self, index):
    return self.q[index]

  def __iter__(self):
    return iter(self.q)

# pylint: disable=line-too-long
# The Code is taken from `https://github.com/takoika/PrioritizedExperienceReplay/blob/master/sum_tree.py`
# pylint: enable=line-too-long


class SumTree(object):
  """SumTree for Prioritized Experience Replay"""

  def __init__(self, max_size=10000000):
    self.max_size = max_size
    self.tree_level = math.ceil(math.log(max_size + 1, 2)) + 1
    self.tree_size = 2 ** self.tree_level - 1
    self.tree = [0 for _ in range(self.tree_size)]
    self.data = [None for _ in range(self.max_size)]
    self.size = 0
    self.cursor = 0

  def add(self, contents, value):
    index = self.cursor
    self.cursor = (self.cursor + 1) % self.max_size
    self.size = min(self.size + 1, self.max_size)

    self.data[index] = contents
    self.val_update(index, value)

  def get_val(self, index):
    tree_index = 2 ** (self.tree_level - 1) - 1 + index
    return self.tree[tree_index]

  def val_update(self, index, value):
    tree_index = 2 ** (self.tree_level - 1) - 1 + index
    diff = value - self.tree[tree_index]
    self.reconstruct(tree_index, diff)

  def reconstruct(self, tindex, diff):
    self.tree[tindex] += diff
    if not tindex == 0:
      tindex = int((tindex - 1) / 2)
      self.reconstruct(tindex, diff)

  def find(self, value, norm=True):
    if norm:
      value *= self.tree[0]
    return self._find(value, 0)

  def _find(self, value, index):
    if 2 ** (self.tree_level - 1) - 1 <= index:
      return self.data[
          index -
          (2 ** (self.tree_level - 1) -
           1)], self.tree[index], index - (2 ** (self.tree_level - 1) - 1)

    left = self.tree[2 * index + 1]

    if value <= left:
      return self._find(value, 2 * index + 1)
    else:
      return self._find(value - left, 2 * (index + 1))

  def print_tree(self):
    for k in range(1, self.tree_level + 1):
      for j in range(2 ** (k - 1) - 1, 2 ** k - 1):
        print(self.tree[j], end=" ")
      print()

  def filled_size(self):
    return self.size

  def __repr__(self) -> str:
    return f"{self.__class__.__name__}({", ".join(repr(e) for e in self.data)})"

  def __str__(self) -> str:
    return f"{self.__class__.__name__}({", ".join(str(e) for e in self.data)})"

  def __contain__(self, contents):
    return contents in self.data

  def __getitem__(self, index):
    return self.data[index]

  def __iter__(self):
    return iter(self.data)


# pylint: disable=line-too-long
# The Code is modified from `https://github.com/takoika/PrioritizedExperienceReplay/blob/master/proportional.py`
# pylint: enable=line-too-long


class ProportionalPrioritizedReplayBuffer():
  """Proportional Prioritized Replay Buffer for RL training"""

  def __init__(self, max_size=None, alpha=0.6, beta=0.4, epsilon=0.01) -> None:
    self.alpha = alpha
    self.beta = beta
    self.epsilon = epsilon
    self.max_size = 1000000 if max_size is None else max_size
    self.tree = SumTree(self.max_size)
    self.sample_weights: list = []
    self.sample_indices: list = []

  def enqueue(self, sample):
    if not self.tree.filled_size():
      return self._enqueue(sample, 1.0)
    priority = max(self.tree.tree[0:self.tree.size])
    return self._enqueue(sample, priority)

  def _enqueue(self, sample, priority):
    self.tree.add(sample, priority ** self.alpha)

  def sample_from(self, sample_ratio=None, num_samples=1, **kwargs):
    """Sample a batch of experiences from the replay buffer"""
    _ = kwargs
    if not self.tree.filled_size():
      raise ValueError("There is not enough samples in the buffer.")

    if sample_ratio:
      num_samples = max(int(self.tree.filled_size() * sample_ratio), 1)

    if self.tree.filled_size() < num_samples:
      return [], [], []

    out = []
    indices = []
    weights = []
    priorities = []
    for _ in range(num_samples):
      r = random.random()
      data, priority, index = self.tree.find(r)
      priorities.append(priority)
      weights.append((1. / self.max_size /
                      priority) ** self.beta if priority > 1e-16 else 0)
      indices.append(index)
      out.append(data)
      self._priority_update([index], [0])  # To avoid duplicating

    self._priority_update(indices, priorities)  # Revert priorities

    weights = [w / max(weights) for w in weights]  # Normalize for stability

    self.sample_weights = weights
    self.sample_indices = indices

    return out

  def update(self, error):
    return self._priority_update(self.sample_indices, error)

  def _priority_update(self, indices, priorities):
    """ The methods update samples"s priority.

      Parameters
      ----------
      indices :
          list of sample indices
      """
    for i, p in zip(indices, priorities):
      self.tree.val_update(i, p ** self.alpha)

  def reset_alpha(self, alpha):
    """ Reset a exponent alpha.

      Parameters
      ----------
      alpha : float
      """
    self.alpha, old_alpha = alpha, self.alpha
    priorities = [
        self.tree.get_val(i) ** -old_alpha
        for i in range(self.tree.filled_size())
    ]
    self._priority_update(range(self.tree.filled_size()), priorities)

  def __repr__(self) -> str:
    return f"{self.__class__.__name__}({", ".join(repr(e) for e in self.tree)})"

  def __str__(self) -> str:
    return f"{self.__class__.__name__}({", ".join(str(e) for e in self.tree)})"

  def __contain__(self, e):
    return e in self.tree

  def __getitem__(self, index):
    return self.tree[index]

  def __iter__(self):
    return iter(self.tree)

  def __len__(self):
    return self.tree.filled_size()
