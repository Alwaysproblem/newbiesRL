"""Helper functions for Advantages calculation."""
import torch


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
