"""SAC implementation with pytorch."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from util.buffer import ReplayBuffer
from util.agent import Agent
from util.buffer import Experience
from util.dist import SquashedNormal


class Actor(nn.Module):
  """ Actor (Policy) Network."""

  def __init__(
      self,
      state_dim,
      action_space,
      seed=0,
      fc1_unit=64,
      fc2_unit=64,
      max_action=1,
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
    super().__init__()  ## calls __init__ method of nn.Module class
    self.seed = torch.manual_seed(seed)
    self.fc1 = nn.Linear(state_dim, fc1_unit)
    self.fc1_ln = nn.LayerNorm(fc1_unit)
    self.fc2 = nn.Linear(fc1_unit, fc2_unit)
    self.fc2_ln = nn.LayerNorm(fc2_unit)
    self.fc_mu_policy = nn.Linear(fc2_unit, action_space)
    self.fc_std_policy = nn.Linear(fc2_unit, action_space)
    self.max_action = max_action

    nn.init.orthogonal_(self.fc1.weight, gain=init_weight_gain)
    nn.init.orthogonal_(self.fc2.weight, gain=init_weight_gain)
    nn.init.uniform_(
        self.fc_mu_policy.weight, -init_policy_weight_gain,
        init_policy_weight_gain
    )
    nn.init.uniform_(
        self.fc_std_policy.weight, -init_policy_weight_gain,
        init_policy_weight_gain
    )

    nn.init.constant_(self.fc1.bias, init_bias)
    nn.init.constant_(self.fc2.bias, init_bias)
    nn.init.constant_(self.fc_mu_policy.bias, init_bias)
    nn.init.constant_(self.fc_std_policy.bias, init_bias)

  def forward(self, x):
    """
        Build a network that maps state -> action values.
        """
    x = F.relu(self.fc1_ln(self.fc1(x)))
    x = F.relu(self.fc2_ln(self.fc2(x)))
    mu = self.fc_mu_policy(x)
    log_std = self.fc_std_policy(x)
    return mu, log_std


class Critic(nn.Module):
  """ Critic (Policy) Model."""

  def __init__(
      self,
      state_dim,
      action_space=1,
      seed=0,
      fc1_unit=64,
      fc2_unit=64,
      init_weight_gain=np.sqrt(2),
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
    super().__init__()  ## calls __init__ method of nn.Module class
    self.seed = torch.manual_seed(seed)
    self.fc1 = nn.Linear(state_dim + action_space, fc1_unit)
    self.fc1_ln = nn.LayerNorm(fc1_unit)
    self.fc2 = nn.Linear(fc1_unit, fc2_unit)
    self.fc2_ln = nn.LayerNorm(fc2_unit)
    self.fc3 = nn.Linear(fc2_unit, 1)

    nn.init.orthogonal_(self.fc1.weight, gain=init_weight_gain)
    nn.init.orthogonal_(self.fc2.weight, gain=init_weight_gain)

    nn.init.constant_(self.fc1.bias, init_bias)
    nn.init.constant_(self.fc2.bias, init_bias)

  def forward(self, x, y):
    """
        Build a network that maps state -> action values.
        """
    x = torch.concat([x, y], dim=1)
    x = F.relu(self.fc1_ln(self.fc1(x)))
    x = F.relu(self.fc2_ln(self.fc2(x)))
    return self.fc3(x)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SACAgent(Agent):
  """Interacts with and learns form environment."""

  def __init__(
      self,
      state_dims,
      action_space,
      gamma=0.99,
      lr_actor=0.001,
      lr_critic=0.001,
      lr_alpha=0.001,
      batch_size=64,
      epsilon=0.01,
      mem_size=None,
      forget_experience=True,
      update_tau=0.5,
      init_alpha=0.1,
      learnable_alpha=True,
      seed=0,
  ):

    self.state_dims = state_dims.shape[0]
    self.action_space_env = action_space
    self.action_space = action_space.shape[0]
    self.gamma = gamma
    self.batch_size = batch_size
    self.epsilon = epsilon
    self.seed = np.random.seed(seed)
    self.lr_actor = lr_actor
    self.lr_critic = lr_critic
    self.log_alpha = torch.tensor(np.log(init_alpha)).to(device)
    self.log_alpha.requires_grad = True
    # set target entropy to - |A|
    self.target_entropy = -self.action_space
    self.lr_alpha = lr_alpha
    self.learnable_alpha = learnable_alpha

    self.update_tau = update_tau

    # Theta 1 network
    self.actor = Actor(
        self.state_dims,
        self.action_space,
        max_action=self.action_space_env.high[0]
    ).to(device)

    # Theta 1 Critic network
    self.critic = Critic(self.state_dims, self.action_space).to(device)
    self.critic_target = Critic(self.state_dims, self.action_space).to(device)
    self.critic_target.load_state_dict(self.critic.state_dict())

    # Theta 2 Critic network
    self.critic_1 = Critic(self.state_dims, self.action_space).to(device)
    self.critic_target_1 = Critic(self.state_dims, self.action_space).to(device)
    self.critic_target_1.load_state_dict(self.critic_1.state_dict())

    self.actor_optimizer = torch.optim.Adam(
        self.actor.parameters(), lr=self.lr_actor
    )
    self.critic_optimizer = torch.optim.Adam([
        *self.critic.parameters(), *self.critic_1.parameters()
    ],
                                             lr=self.lr_critic)
    self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr_alpha)

    # Replay memory
    self.memory = ReplayBuffer(max_size=mem_size)
    self.forget_experience = forget_experience

    self.squashed_normal = SquashedNormal()

  def learn(self, iteration):
    if len(self.memory) > self.batch_size:
      for _ in range(iteration):
        experience = self.memory.sample_from(num_samples=self.batch_size)
        self._learn(experience)

  def action(self, state, mode="eval"):
    if mode == "train":
      self.actor.train()
    else:
      self.actor.eval()

    mu, log_std = self.actor.forward(state)

    dist = self.squashed_normal(mu, log_std.exp())
    action = dist.sample()

    return action[0].cpu().data.numpy(), dist

  def take_action(self, state, explore=False):
    """Returns action for given state as per current policy
        Params
        =======
            state (array_like): current state
            epsilon (float): epsilon, for epsilon-greedy action selection
        """
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)

    with torch.no_grad():
      action_values, *_ = self.action(state)

    # Clip the output according to the action space of the env
    action_values = np.clip(
        action_values, self.action_space_env.low[0],
        self.action_space_env.high[0]
    )
    return action_values

  def remember(self, scenario: Experience):
    self.memory.enqueue(scenario)

  def _train_critic(self, states, actions, rewards, next_states, terminate):
    # log π(at+1|st+1)
    _, dist = self.action(next_states, mode="train")
    next_action = dist.rsample()
    log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)

    # Compute the Q value with TD3 trick
    target_q = self.critic_target.forward(next_states, next_action)
    target_q_1 = self.critic_target_1.forward(next_states, next_action)

    min_target_q_value = torch.min(target_q, target_q_1)

    # Compute the target Value with
    # V (st+1) = E at∼π [Q(st+1, at+1) − α log π(at+1|st+1)]
    target_v = min_target_q_value - self.log_alpha.exp().detach() * log_prob
    target_q = rewards + ((1 - terminate) * self.gamma * target_v).detach()

    # Get current Q estimate
    current_q = self.critic.forward(states, actions)
    current_q_1 = self.critic_1.forward(states, actions)

    # Compute critic loss
    critic_loss = F.mse_loss(current_q,
                             target_q) + F.mse_loss(current_q_1, target_q)

    # Optimize the critic
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()

  def _train_actor(self, states):
    # log π(at|st)
    _, dist = self.action(states, mode="train")
    action = dist.rsample()
    log_prob = dist.log_prob(action).sum(-1, keepdim=True)

    # Compute the Q value with TD3 trick
    target_q = self.critic.forward(states, action)
    target_q_1 = self.critic_1.forward(states, action)

    min_target_q_value = torch.min(target_q, target_q_1)

    # Jπ(φ)=Est∼D[E at∼πφ [αlog(πφ(at|st))−Qθ(st, at)]]
    actor_loss = (
        self.log_alpha.exp().detach() * log_prob - min_target_q_value
    ).mean()

    # Optimize the actor
    self.actor_optimizer.zero_grad()
    actor_loss.backward()
    self.actor_optimizer.step()

    if self.learnable_alpha:
      self.alpha_optimizer.zero_grad()
      alpha_loss = (
          self.log_alpha.exp() * (-log_prob - self.target_entropy).detach()
      ).mean()
      alpha_loss.backward()
      self.alpha_optimizer.step()

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

    self.critic.train()
    self.critic_target.eval()

    self.critic_1.train()
    self.critic_target_1.eval()

    self._train_critic(states, actions, rewards, next_states, terminate)
    self._train_actor(states)

    self.update_critic_target_network()
    # self.update_critic_target_1_network()

  def soft_update(self, local_model, target_model):
    """
      Soft update model parameters.
      θ_target = τ * θ_local + (1 - τ) * θ_target
      Token from
      https://github.com/udacity/deep-reinforcement-learning/blob/master/dqn/exercise/dqn_agent.py
    """
    for target_param, local_param in zip(
        target_model.parameters(), local_model.parameters()
    ):
      target_param.data.copy_(
          self.update_tau * local_param.data +
          (1.0 - self.update_tau) * target_param.data
      )

  def update_actor_target_network(self):
    self.soft_update(self.actor, self.actor_target)

  def update_critic_target_network(self):
    self.soft_update(self.critic, self.critic_target)

  def update_critic_target_1_network(self):
    self.soft_update(self.critic_1, self.critic_target_1)
