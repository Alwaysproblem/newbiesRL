"""The basic frame for Agent"""


class Agent():
  """The basic class for agent"""

  def __call__(self, *args, **kwds):
    return self.call(*args, **kwds)

  def call(self, state):
    raise NotImplementedError()

  def explore(self, state):
    raise NotImplementedError()

  def learn(self):
    raise NotImplementedError()

  def learn_from(self, other):
    raise NotImplementedError()
