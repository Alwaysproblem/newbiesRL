"""Distribution Q learning implementation with pytorch."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from util.buffer import ReplayBuffer
from util.agent import Agent
from util.buffer import Experience


class Q(nn.Module):
  """ Actor (Policy) Model."""
  def __init__(
      self,
      state_dim,
      action_space,
      n_atoms,
      seed=0,
      hidden_size=None,
      init_weight_gain=np.sqrt(2),
      init_policy_weight_gain=1,
      init_bias=0
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
    super().__init__()
    self.action_space = action_space
    self.n_atoms = n_atoms
    self.seed = torch.manual_seed(seed)
    self.hidden_size = (100, 100, 100) if not hidden_size else hidden_size

    def init_weights(m):
      if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=init_weight_gain)
        nn.init.constant_(m.bias, init_bias)

    # note:  The self.hidden_layers attribute is defined as a list of lists,
    # note:  but it should be a list of `nn.Sequential` objects.
    # note:  You can fix this by using `nn.Sequential` to define each layer.
    # note:  After using `nn.Sequential`, you need to define a list with
    # note:  `nn.ModuleList` to construct the model graph.
    self.hidden_layers = nn.ModuleList([
        nn.Sequential(
            nn.Linear(in_size, out_size), nn.LayerNorm(out_size), nn.ReLU()
        ) for in_size, out_size in zip((state_dim, ) +
                                       self.hidden_size, self.hidden_size)
    ])
    self.hidden_layers.apply(init_weights)

    def init_output_weights(m):
      if isinstance(m, nn.Linear):
        # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # nn.init.constant_(m.bias, init_bias)
        nn.init.orthogonal_(m.weight, gain=init_weight_gain)
        nn.init.constant_(m.bias, init_bias)

    # self.output_layer = nn.Linear(self.hidden_size[-1], action_space * n_atoms)
    self.output_layers = nn.ModuleList([
        nn.Sequential(
            nn.Linear(self.hidden_size[-1], n_atoms), nn.LayerNorm(n_atoms), nn.Softmax(dim=-1)
        ) for _ in range(action_space)
    ])

    self.output_layers.apply(init_output_weights)

  def forward(self, state):
    x = state
    for hidden_layer in self.hidden_layers:
      x = hidden_layer(x)
    out = torch.concat([torch.unsqueeze(output_layer(x), dim=1) for output_layer in self.output_layers], dim=1)
    # x = self.output_layer(x)
    # x = torch.reshape(x, (-1, self.action_space, self.n_atoms))
    # x = F.softmax(x, dim=-1)
    return out


# device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class C51Agent(Agent):
  """Interacts with and learns form environment."""
  def __init__(
      self,
      state_dims,
      action_space,
      n_atoms,
      v_min,
      v_max,
      gamma=0.99,
      lr=0.001,
      batch_size=64,
      epsilon=0.01,
      mem_size=None,
      forget_experience=True,
      sample_ratio=None,
      seed=0,
  ):

    self.state_dims = state_dims
    self.action_space = action_space
    self.gamma = gamma
    self.batch_size = batch_size
    self.epsilon = epsilon
    self.seed = torch.manual_seed(seed)
    self.sample_ratio = sample_ratio
    self.n_atoms = n_atoms
    self.v_min = v_min
    self.v_max = v_max
    self.delta = (v_max - v_min) / n_atoms
    self.atoms = torch.arange(self.v_min, self.v_max, self.delta).to(device)

    #Q- Network
    self.qnetwork_local = Q(
        state_dim=state_dims, action_space=action_space, n_atoms=n_atoms
    ).to(device)
    self.qnetwork_target = Q(
        state_dim=state_dims, action_space=action_space, n_atoms=n_atoms
    ).to(device)
    self.optimizer = torch.optim.Adam(self.qnetwork_local.parameters(), lr=lr)

    # Replay memory
    self.memory = ReplayBuffer(max_size=mem_size)

    self.forget_experience = forget_experience
    self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

  def learn(self, iteration):
    if len(self.memory) > self.batch_size:
      for _ in range(iteration):
        experience = self.memory.sample_from(num_samples=self.batch_size)
        self._learn(experience)

  def action(self, state):
    # possiblities: (batch, action_dim, n_atoms)
    possiblities = self.qnetwork_local.forward(state)
    # z: (n_atoms, 1)
    z = self.atoms.unsqueeze(1)
    #  q_value = (batch, action_dim, n_atoms) x (n_atoms, 1)
    #  = (batch, action_dim, 1)
    q_value = possiblities @ z
    # a_star = (batch, 1)
    a_star = torch.argmax(q_value, dim=1)
    return a_star

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
      action_values = self.action(state)
    self.qnetwork_local.train()

    #Epsilon -greedy action selction
    if np.random.random() > epsilon:
      return action_values.squeeze().cpu().data.numpy()
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

    m = self.calc_q_target(rewards, next_states, terminate).detach()

    pos_s = self.qnetwork_local.forward(states)

    pos_s_a = torch.sum(
        F.one_hot(actions, self.action_space).unsqueeze(-1).repeat(
            (1, 1, 1, self.n_atoms)
        ).squeeze(1) * pos_s,
        dim=-2
    )

    loss = -torch.mean(torch.sum(m * (pos_s_a + 1e-8).log(), dim=-1)).to(device)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

  def calc_q_target(self, rewards, next_states, terminate):
    """
    Calculate the target distributions for the given experience.
    reference: https://arxiv.org/pdf/1707.06887.pdf
    """
    # possiblities: (batch, action_dim, n_atoms)
    possiblities = self.qnetwork_target.forward(next_states)
    # z: (n_atoms, 1)
    z = self.atoms.unsqueeze(1)
    #  q_value = (batch, action_dim, n_atoms) x (n_atoms, 1)
    #          = (batch, action_dim, 1)
    # q_value = ∑ᵢzᵢpᵢ(xₜ₊₁, a)
    q_value = possiblities @ z
    # a_star = (batch, 1)
    # a* <- argmaxₐQ(xₜ₊₁, a)
    a_star = torch.argmax(q_value, dim=1)

    # tau_z = (batch, n_atom)
    # tau_z = clip(rₜ + γₜ * zⱼ, v_min, v_max)
    tau_z = torch.clamp(
        rewards + self.gamma * (1 - terminate) @ z.T,
        min=self.v_min,
        max=self.v_max
    )
    # bⱼ ∈ [0, N - 1], shape: (batch, n_atom)
    b_j = (tau_z - self.v_min) / self.delta
    l = torch.floor(b_j).clamp(max=self.n_atoms - 1).long()
    u = torch.ceil(b_j).clamp(max=self.n_atoms - 1
                              ).long()  # Prevent out of bounds
    # m: shape (batch, n_atoms)
    m = torch.zeros(size=(
        next_states.shape[0],
        self.n_atoms,
    )).to(device)

    # this can be implemented by gather op
    # pᵢ(xₜ₊₁, a) shape: (batch, n_atoms)
    p_j = torch.sum(
        F.one_hot(a_star, self.action_space).unsqueeze(-1).repeat(
            (1, 1, 1, self.n_atoms)
        ).squeeze(1) * possiblities,
        dim=-2
    )
    delta_m_l = p_j * (u - b_j)
    delta_m_u = p_j * (b_j - l)

    # mₗ ← mₗ + pⱼ(xt+1, a*)(u − bj )
    m.scatter_add_(1, l, delta_m_l)
    # mᵤ ← mᵤ + pⱼ(xt+1, a*)(bj − l)
    m.scatter_add_(1, u, delta_m_u)
    return m

  def update_targe_q(self):
    self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
