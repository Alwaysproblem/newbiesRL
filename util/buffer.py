"""The buffer protocol described here"""
from collections import deque
from copy import deepcopy
import warnings
import random
import numpy as np
from util.tree import SumTree


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
  """Replay Buffer for off-policy training"""

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

  def update(self, error):
    pass

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
# # The Code is taken from https://github.com/Howuhh/prioritized_experience_replay/blob/main/memory/buffer.py
class ProportionalPrioritizedReplayBuffer:
  """Proportional Prioritized ReplayBuffer for off-policy training"""

  def __init__(self, max_size=None, eps=1e-2, alpha=0.1, beta=0.1):
    self.size = max_size if max_size else 1000000
    self.tree = SumTree(size=self.size)

    # PER params
    self.eps = eps  # minimal priority, prevents zero probabilities
    self.alpha = alpha  # determines how much prioritization is used, α = 0 corresponding to the uniform case
    self.beta = beta  # determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
    self.max_priority = eps  # priority for new samples, init as eps

    self.count = 0
    self.real_size = 0
    self.sample_weights = []
    self.sample_indices = []

  def enqueue(self, sample):

    # store transition index with maximum priority in sum tree
    self.tree.add(self.max_priority, sample)

    # update counters
    self.count = (self.count + 1) % self.size
    self.real_size = min(self.size, self.real_size + 1)

  def sample_from(self, sample_ratio=None, num_samples=1, **kwargs):
    _ = kwargs
    if sample_ratio:
      num_samples = max(int(self.real_size * sample_ratio), 1)

    if self.real_size <= num_samples:
      return []

    samples, indices = [], []

    # priorities = torch.empty(num_samples, 1, dtype=torch.float)
    priorities = np.zeros(num_samples, dtype=np.float32)

    # To sample a minibatch of size k, the range [0, p_total] is divided equally into k ranges.
    # Next, a value is uniformly sampled from each range. Finally the transitions that correspond
    # to each of these sampled values are retrieved from the tree. (Appendix B.2.1, Proportional prioritization)
    segment = self.tree.total / num_samples
    for i in range(num_samples):
      a, b = segment * i, segment * (i + 1)

      cumsum = random.uniform(a, b)
      # sample_idx is a sample index in buffer, needed further to sample actual transitions
      # index is a index of a sample in the tree, needed further to update priorities
      index, priority, sample_idx = self.tree.get(cumsum)

      priorities[i] = priority
      indices.append(index)
      samples.append(sample_idx)

    # Concretely, we define the probability of sampling transition i as P(i) = p_i^α / \sum_{k} p_k^α
    # where p_i > 0 is the priority of transition i. (Section 3.3)
    probs = priorities / self.tree.total

    # The estimation of the expected value with stochastic updates relies on those updates corresponding
    # to the same distribution as its expectation. Prioritized replay introduces bias because it changes this
    # distribution in an uncontrolled fashion, and therefore changes the solution that the estimates will
    # converge to (even if the policy and state distribution are fixed). We can correct this bias by using
    # importance-sampling (IS) weights w_i = (1/N * 1/P(i))^β that fully compensates for the non-uniform
    # probabilities P(i) if β = 1. These weights can be folded into the Q-learning update by using w_i * δ_i
    # instead of δ_i (this is thus weighted IS, not ordinary IS, see e.g. Mahmood et al., 2014).
    # For stability reasons, we always normalize weights by 1/maxi wi so that they only scale the
    # update downwards (Section 3.4, first paragraph)
    weights = (self.real_size * probs) ** -self.beta

    # As mentioned in Section 3.4, whenever importance sampling is used, all weights w_i were scaled
    # so that max_i w_i = 1. We found that this worked better in practice as it kept all weights
    # within a reasonable range, avoiding the possibility of extremely large updates. (Appendix B.2.1, Proportional prioritization)
    weights = weights / weights.max()

    self.sample_weights = weights
    self.sample_indices = indices

    return samples

  def update_priorities(self, data_idxs, priorities):
    for data_idx, priority in zip(data_idxs, priorities):
      # The first variant we consider is the direct, proportional prioritization where p_i = |δ_i| + eps,
      # where eps is a small positive constant that prevents the edge-case of transitions not being
      # revisited once their error is zero. (Section 3.3)
      priority = (priority + self.eps) ** self.alpha

      self.tree.update(data_idx, priority)
      self.max_priority = max(self.max_priority, priority)

  def update(self, error):
    self.update_priorities(self.sample_indices, error)

  def __repr__(self) -> str:
    return f"{self.__class__.__name__}({self.tree.__repr__()})"

  def __str__(self) -> str:
    return f"{self.__class__.__name__}({self.tree.__repr__()})"

  def __contain__(self, e):
    return e in self.tree

  def __getitem__(self, index):
    return self.tree[index]

  def __iter__(self):
    return iter(self.tree)

  def __len__(self):
    return self.tree.real_size
