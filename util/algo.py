"""Helper functions for Advantages calculation."""
import torch


def standardize(v, normeps=1e-8):
  """Method to standardize a rank-1 np array."""
  assert len(v) > 1, "Cannot standardize vector of size 1"
  v_std = (v - v.mean()) / (v.std() + normeps)
  return v_std


def scale_up_values(v, mean=0, std=1, norm_factor=1):
  return v / norm_factor * std + mean


def scale_down_values(v, mean=0, std=1, norm_factor=1, normeps=1e-8):
  return norm_factor * (v - mean) / (std + normeps)


def calc_nstep_return(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    next_v_pred: torch.Tensor,
    gamma=0.99,
    n_steps=5
):
  T = len(rewards)  #pylint: disable=invalid-name
  device = rewards.device
  rets = torch.zeros_like(rewards, device=device)
  _ = 1 - dones

  for i in range(T):
    # we generate the vector like `gamma = [[γ⁰, γ¹, γ² ...γⁿ]]`
    # and gamma x reward (vector) to obtain the value for each timestamp.
    # There are a few items to make it to N
    # and we will take account all the items.
    rets[i] = torch.unsqueeze(
        gamma ** torch.arange(len(rewards[i:min(n_steps + i, T)])).to(device),
        dim=0
    ) @ rewards[i:min(n_steps + i, T)]

  if T > n_steps:
    # [[γ⁰, γ¹, γ² ...γⁿ]] x reward.T + γⁿ⁺¹ * V(sₜ₊ₙ₊₁)
    value_n_steps = gamma ** n_steps * next_v_pred[n_steps:]
    rets = torch.cat([
        value_n_steps,
        torch.zeros(size=(n_steps, 1), device=device)
    ]) + rets

  return rets


def calc_gaes(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    v_preds: torch.Tensor,
    gamma=0.99,
    gae_lambda=0.95
):
  # GAE = ∑ₗ (γλ)ˡδₜ₊ₗ
  # δₜ₊ₗ = rₜ + γV(sₜ₊₁) − V(sₜ)
  T = len(rewards)  # pylint: disable=invalid-name
  device = rewards.device
  gaes = torch.zeros_like(rewards, device=device)
  future_gae = torch.tensor(0.0, dtype=rewards.dtype, device=device)
  not_dones = 1 - dones  # to reset at episode boundary by multiplying 0
  deltas = rewards + gamma * v_preds[1:] * not_dones - v_preds[:-1]
  coef = gamma * gae_lambda
  for t in reversed(range(T)):
    gaes[t] = future_gae = deltas[t] + coef * not_dones[t] * future_gae
  return gaes


def gumbel_loss(pred, label, beta, clip):
  """
    Gumbel loss function

    Describe in Appendix D.3 of

    https://arxiv.org/pdf/2301.02328.pdf

    Token from
    https://github.com/Div99/XQL/blob/dff09afb893fe782be259c2420903f8dfb50ef2c/online/research/algs/gumbel_sac.py#L10)
  """
  assert pred.shape == label.shape, "Shapes were incorrect"
  z = (label - pred) / beta
  if clip is not None:
    z = torch.clamp(z, -clip, clip)
  loss = torch.exp(z) - z - 1
  return loss.mean()


def gumbel_rescale_loss(pred, label, beta, clip):
  """
    Gumbel rescale loss function

    Describe in Appendix D.3 (NUMERIC STABILITY) of

    https://arxiv.org/pdf/2301.02328.pdf

    Token from
    https://github.com/Div99/XQL/blob/dff09afb893fe782be259c2420903f8dfb50ef2c/online/research/algs/gumbel_sac.py#L18)
  """
  assert pred.shape == label.shape, "Shapes were incorrect"
  z = (label - pred) / beta
  if clip is not None:
    z = torch.clamp(z, -clip, clip)
  max_z = torch.max(z)
  max_z = torch.where(
      max_z < -1.0, torch.tensor(-1.0, dtype=torch.float, device=max_z.device),
      max_z
  )
  max_z = max_z.detach()  # Detach the gradients
  loss = torch.exp(z - max_z) - z * torch.exp(-max_z) - torch.exp(-max_z)
  return loss.mean()
