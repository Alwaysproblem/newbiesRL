"""TD3 implementation with pytorch."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from util.buffer import ReplayBuffer
from util.agent import Agent
from util.buffer import Experience


class OUNoise(object):
  # pylint: disable=line-too-long
  """
    Taken from https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
    """

  def __init__(
      self,
      action_space,
      mu=0.0,
      theta=0.15,
      max_sigma=0.3,
      min_sigma=0.3,
      decay_period=100000
  ):
    self.mu = mu
    self.theta = theta
    self.sigma = max_sigma
    self.max_sigma = max_sigma
    self.min_sigma = min_sigma
    self.decay_period = decay_period
    self.action_dim = action_space.shape[0]
    self.low = action_space.low
    self.high = action_space.high
    self.reset()

  def reset(self):
    self.state = np.ones(self.action_dim) * self.mu

  def evolve_state(self):
    x = self.state
    dx = self.theta * (self.mu -
                       x) + self.sigma * np.random.randn(self.action_dim)
    self.state = x + dx
    return self.state

  def get_action(self, action, t=0):
    ou_state = self.evolve_state()
    self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma
                                   ) * min(1.0, t / self.decay_period)
    return np.clip(action + ou_state, self.low, self.high)


class Actor(nn.Module):
  """ Actor (Policy) Model."""

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
    self.fc_policy = nn.Linear(fc2_unit, action_space)
    self.max_action = max_action

    nn.init.orthogonal_(self.fc1.weight, gain=init_weight_gain)
    nn.init.orthogonal_(self.fc2.weight, gain=init_weight_gain)
    nn.init.uniform_(
        self.fc_policy.weight, -init_policy_weight_gain, init_policy_weight_gain
    )

    nn.init.constant_(self.fc1.bias, init_bias)
    nn.init.constant_(self.fc2.bias, init_bias)
    nn.init.constant_(self.fc_policy.bias, init_bias)

  def forward(self, x):
    """
        Build a network that maps state -> action values.
        """
    x = F.relu(self.fc1_ln(self.fc1(x)))
    x = F.relu(self.fc2_ln(self.fc2(x)))
    pi = self.max_action * torch.tanh(self.fc_policy(x))
    return pi


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


class TD3Agent(Agent):
  """Interacts with and learns form environment."""

  def __init__(
      self,
      state_dims,
      action_space,
      gamma=0.99,
      lr_actor=0.001,
      lr_critic=0.001,
      batch_size=64,
      epsilon=0.01,
      mem_size=None,
      forget_experience=True,
      update_tau=0.5,
      n_steps=0,
      gae_lambda=None,
      beta=0,
      seed=0,
      mu=0.0,
      theta=0.15,
      max_sigma=0.3,
      min_sigma=0.3,
      decay_period=100000,
      value_noise_clip=0.5,
      value_noise_sigma=0.5
  ):

    self.state_dims = state_dims.shape[0]
    self.action_space_env = action_space
    self.action_space = action_space.shape[0]
    self.gamma = gamma
    self.batch_size = batch_size
    self.epsilon = epsilon
    self.seed = np.random.seed(seed)
    self.n_steps = n_steps
    self.gae_lambda = gae_lambda
    self.lr_actor = lr_actor
    self.lr_critic = lr_critic
    self.beta = beta
    self.noise = OUNoise(
        action_space,
        mu=mu,
        theta=theta,
        max_sigma=max_sigma,
        min_sigma=min_sigma,
        decay_period=decay_period,
    )
    self.update_tau = update_tau
    self.value_noise_clip = value_noise_clip
    self.value_noise_sigma = value_noise_sigma

    # Theta 1 network
    self.actor = Actor(
        self.state_dims,
        self.action_space,
        max_action=self.action_space_env.high[0]
    ).to(device)
    self.actor_target = Actor(
        self.state_dims,
        self.action_space,
        max_action=self.action_space_env.high[0]
    ).to(device)
    self.actor_target.load_state_dict(self.actor.state_dict())

    self.actor_optimizer = torch.optim.Adam(
        self.actor.parameters(), lr=self.lr_actor
    )

    # Theta 1 Critic network
    self.critic = Critic(self.state_dims, self.action_space).to(device)
    self.critic_target = Critic(self.state_dims, self.action_space).to(device)
    self.critic_target.load_state_dict(self.critic.state_dict())

    # Theta 2 Critic network
    self.critic_1 = Critic(self.state_dims, self.action_space).to(device)
    self.critic_target_1 = Critic(self.state_dims, self.action_space).to(device)
    self.critic_target_1.load_state_dict(self.critic_1.state_dict())

    self.critic_optimizer = torch.optim.Adam(
        self.critic.parameters(), lr=self.lr_critic
    )

    # Replay memory
    self.memory = ReplayBuffer(max_size=mem_size)

    self.forget_experience = forget_experience

    self.val_loss = nn.MSELoss()
    self.val_1_loss = nn.MSELoss()
    self.policy_loss = nn.MSELoss()

  def learn(self, iteration):
    if len(self.memory) > self.batch_size:
      for _ in range(iteration):
        experience = self.memory.sample_from(num_samples=self.batch_size)
        self._learn(experience)

  def action(self, state, mode="train"):
    if mode == "train":
      self.actor.train()
    else:
      self.actor.eval()

    with torch.no_grad():
      action = self.actor.forward(state)
    return action.cpu().data.numpy()

  def take_action(self, state, explore=False, step=0):
    """Returns action for given state as per current policy
        Params
        =======
            state (array_like): current state
            epsilon (float): epsilon, for epsilon-greedy action selection
        """
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    action_values = self.action(state=state, mode="eval").squeeze(0)
    if explore:
      action_values = self.noise.get_action(action_values, step)

    # Clip the output according to the action space of the env
    action_values = np.clip(
        action_values, self.action_space_env.low[0],
        self.action_space_env.high[0]
    )
    return action_values

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

    self.critic.train()
    self.critic_target.eval()
    self.critic_1.train()
    self.critic_target_1.eval()
    self.actor_target.eval()

    # noise

    noise = torch.clamp(
        torch.normal(mean=0.0, std=self.value_noise_sigma, size=actions.size()),
        -self.value_noise_clip, self.value_noise_clip
    ).to(device)

    # Compute the target Q value
    target_q = self.critic_target.forward(
        next_states,
        self.actor_target.forward(next_states) + noise
    )
    target_q_1 = self.critic_target_1.forward(
        next_states,
        self.actor_target.forward(next_states) + noise
    )

    min_target_q_value = torch.min(
        torch.cat((target_q, target_q_1), dim=1), dim=1
    ).values.unsqueeze(dim=1)

    target_q = rewards + ((1 - terminate) * self.gamma *
                          min_target_q_value).detach()

    # Get current Q estimate
    current_q = self.critic.forward(states, actions)

    # Get current Q estimate
    current_q_1 = self.critic_1.forward(states, actions)

    # Compute critic loss
    critic_loss = self.val_loss(current_q, target_q
                                ) + self.val_1_loss(current_q_1, target_q)

    # Optimize the critic
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()

    # Here need to `Delayed policy updates`
    # and I used `policy_freq = 1`
    # Compute actor loss
    actor_loss = -self.critic.forward(states, self.actor.forward(states)).mean()

    # Optimize the actor
    self.actor_optimizer.zero_grad()
    actor_loss.backward()
    self.actor_optimizer.step()

    self.update_actor_target_network()
    self.update_critic_target_network()

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
