"""DQN implementation with pytorch."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from util.agent import Agent
from util.buffer import Experience
from util.buffer import ProportionalPrioritizedReplayBuffer

# class Q(nn.Module):
#   """ Actor (Policy) Model."""

#   def __init__(self, state_dim, action_space, seed=0, hidden_size=None):
#     """
#         Initialize parameters and build model.
#         Params
#         =======
#             state_size (int): Dimension of each state
#             action_size (int): Dimension of each action
#             seed (int): Random seed
#             fc1_unit (int): Number of nodes in first hidden layer
#             fc2_unit (int): Number of nodes in second hidden layer
#         """
#     super().__init__()
#     self.action_space = action_space
#     self.seed = torch.manual_seed(seed)
#     self.hidden_size = (64, 64, 64) if not hidden_size else hidden_size

#     # note:  The self.hidden_layers attribute is defined as a list of lists,
#     # note:  but it should be a list of `nn.Sequential` objects.
#     # note:  You can fix this by using `nn.Sequential` to define each layer.
#     # note:  After using `nn.Sequential`, you need to define a list with
#     # note:  `nn.ModuleList` to construct the model graph.
#     self.hidden_layers = nn.ModuleList([
#         nn.Sequential(nn.Linear(in_size, out_size), nn.ReLU())
#         for in_size, out_size in zip((state_dim, ) +
#                                      self.hidden_size, self.hidden_size)
#     ])
#     self.output_layer = nn.Linear(self.hidden_size[-1], action_space)

#   def forward(self, state):
#     x = state
#     for hidden_layer in self.hidden_layers:
#       x = hidden_layer(x)
#     x = self.output_layer(x)
#     return x


class Q(nn.Module):
  """ Actor (Policy) Model."""

  def __init__(
      self, state_dim, action_space, seed=0, fc1_unit=128, fc2_unit=128
  ):
    """
        Initialize parameters and build model.
        Params
        =======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_unit (int): Number of nodes in first hidden layer
            fc2_unit (int): Number of nodes in second hidden layer
        """
    super().__init__()  ## calls __init__ method of nn.Module class
    self.seed = torch.manual_seed(seed)
    self.fc1 = nn.Linear(state_dim, fc1_unit)
    self.fc2 = nn.Linear(fc1_unit, fc2_unit)
    self.fc3 = nn.Linear(fc2_unit, action_space)

  def forward(self, x):
    """
        Build a network that maps state -> action values.
        """
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    return self.fc3(x)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DQNAgent(Agent):
  """Interacts with and learns form environment."""

  def __init__(
      self,
      state_dims,
      action_space,
      gamma=0.99,
      lr=0.001,
      batch_size=64,
      epsilon=0.01,
      mem_size=None,
      forget_experience=True,
      grad_clip=300,
      sample_ratio=None,
      seed=0
  ):

    self.state_dims = state_dims
    self.action_space = action_space
    self.gamma = gamma
    self.batch_size = batch_size
    self.epsilon = epsilon
    self.seed = np.random.seed(seed)
    self.sample_ratio = sample_ratio
    self.grad_clip = grad_clip

    #Q- Network
    self.qnetwork_local = Q(state_dim=state_dims,
                            action_space=action_space).to(device)
    self.qnetwork_target = Q(state_dim=state_dims,
                             action_space=action_space).to(device)
    self.optimizer = torch.optim.Adam(self.qnetwork_local.parameters(), lr=lr)

    # Replay memory
    self.memory = ProportionalPrioritizedReplayBuffer(max_size=mem_size)
    # self.memory = ReplayBuffer(max_size=mem_size)

    self.forget_experience = forget_experience
    self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
    # self.loss = nn.HuberLoss()
    self.loss = nn.SmoothL1Loss()
    # self.loss = nn.MSELoss()

  def learn(self, iteration):
    if len(self.memory) > self.batch_size:
      for _ in range(iteration):
        experience = self.memory.sample_from(num_samples=self.batch_size)
        self._learn(experience)

  def take_action(self, state, epsilon=0):
    """Returns action for given state as per current policy
        Params
        =======
            state (array_like): current state
            epsilon (float): epsilon, for epsilon-greedy action selection
        """
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    self.qnetwork_local.eval()
    with torch.no_grad():
      action_values = self.qnetwork_local.forward(state)
    self.qnetwork_local.train()

    #Epsilon -greedy action selction
    if np.random.random() > epsilon:
      return np.argmax(action_values.cpu().data.numpy())
    else:
      return np.random.choice(np.arange(self.action_space))

  def remember(self, scenario: Experience):
    self.memory.enqueue(scenario)

  def _learn(self, experiences):
    # pylint: disable=line-too-long
    """Update value parameters using given batch of experience tuples.
        Params
        =======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """

    states = torch.from_numpy(np.vstack([e.state for e in experiences])
                              ).float().to(device)
    actions = torch.from_numpy(np.vstack([e.action for e in experiences])
                               ).long().to(device)
    rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])
                               ).float().to(device)
    next_states = torch.from_numpy(
        np.vstack([e.next_state for e in experiences])
    ).float().to(device)
    terminate = torch.from_numpy(np.vstack([e.done for e in experiences])
                                 ).float().to(device)

    self.qnetwork_local.train()
    self.qnetwork_target.eval()

    # action: [[1]] -> [[0, 1]], onehot encoding for slice the q value.
    actions_onehot = F.one_hot(actions.squeeze(), self.action_space)
    # Q(status) -> [[0.3, 0.4]]
    #                 a1   a2
    # Q(status) * [[0, 1]]:
    #             [[0, 0.4]]
    # after sum:
    #             [[0.4]]
    predicted_targets = torch.sum(
        self.qnetwork_local.forward(states) * actions_onehot,
        axis=1,
        keepdim=True
    )

    with torch.no_grad():
      # r + (1 − done) × γ × max(Q(state))
      labels = rewards + (1 - terminate) * self.gamma * torch.max(
          self.qnetwork_target.forward(next_states).detach(),
          dim=1,
          keepdim=True
      ).values

    self.memory.update(torch.abs(predicted_targets - labels).squeeze().tolist())

    # sample_weight_ratio = 1.0
    sample_weight_ratio = torch.Tensor(
        self.memory.sample_weights,
    ).unsqueeze(1).to(device).detach() * (labels - predicted_targets).detach()

    # Sampled loss function
    loss = self.loss(
        sample_weight_ratio * predicted_targets, sample_weight_ratio * labels
    )
    # loss = self.loss(predicted_targets, labels)
    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(
        self.qnetwork_local.parameters(), self.grad_clip
    )
    self.optimizer.step()

  def update_targe_q(self):
    self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
