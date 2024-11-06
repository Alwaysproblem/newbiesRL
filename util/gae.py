import torch

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
