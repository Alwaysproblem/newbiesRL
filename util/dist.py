# =============================================================================
# MIT License

# Copyright (c) 2023 Reinforcement Learning Evolution Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================
# pylint: disable=line-too-long,unused-argument
# Borrow from `https://github.com/RLE-Foundation/rllte`
"""Distributions for action noise and policy."""

import math
import re
from typing import Any, Tuple, Optional, TypeVar, Union

import numpy as np
import torch as th
from torch.distributions import register_kl
from torch import distributions as pyd
from torch.nn import functional as F
from torch.distributions import Distribution
from torch.distributions.utils import _standard_normal


class BaseDistribution(Distribution):
  """Abstract base class of distributions.
    In rllte, the action noise is implemented as a distribution.
    """

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(validate_args=False)

  def __call__(self, *args, **kwargs) -> Any:
    """Call the distribution."""

  def sample(self, *args, **kwargs) -> th.Tensor:  # type: ignore
    """Generate samples."""


class Bernoulli(BaseDistribution):
  """Bernoulli distribution for sampling actions for 'MultiBinary' tasks."""

  def __init__(self) -> None:
    super().__init__()

  def __call__(self, logits: th.Tensor):
    """Create the distribution.

        Args:
            logits (th.Tensor): The event log probabilities (unnormalized).

        Returns:
            Bernoulli distribution instance.
        """
    self.dist = pyd.Bernoulli(logits=logits)
    return self

  @property
  def probs(self) -> th.Tensor:
    """Return probabilities."""
    return self.dist.probs

  @property
  def logits(self) -> th.Tensor:
    """Returns the unnormalized log probabilities."""
    return self.dist.logits

  def sample(self, sample_shape: th.Size = th.Size()) -> th.Tensor:  # B008
    """Generates a sample_shape shaped sample or sample_shape shaped batch of
            samples if the distribution parameters are batched.

        Args:
            sample_shape (th.Size): The size of the sample to be drawn.

        Returns:
            A sample_shape shaped sample.
        """
    return self.dist.sample()

  def log_prob(self, actions: th.Tensor) -> th.Tensor:
    """Returns the log of the probability density/mass function evaluated at actions.

        Args:
            actions (th.Tensor): The actions to be evaluated.

        Returns:
            The log_prob value.
        """
    return self.dist.log_prob(actions).sum(-1)

  def entropy(self) -> th.Tensor:
    """Returns the Shannon entropy of distribution."""
    return self.dist.entropy().sum(-1)

  @property
  def mode(self) -> th.Tensor:
    """Returns the mode of the distribution."""
    return th.gt(self.dist.probs, 0.5).float()

  @property
  def mean(self) -> th.Tensor:
    """Returns the mean of the distribution."""
    return th.gt(self.dist.probs, 0.5).float()


class Categorical(BaseDistribution):
  """Categorical distribution for sampling actions for 'Discrete' tasks."""

  def __init__(self) -> None:
    super().__init__()

  def __call__(self, logits: th.Tensor):
    """Create the distribution.

        Args:
            logits (th.Tensor): The event log probabilities (unnormalized).

        Returns:
            Categorical distribution instance.
        """
    self.dist = pyd.Categorical(logits=logits)
    return self

  @property
  def probs(self) -> th.Tensor:
    """Return probabilities."""
    return self.dist.probs

  @property
  def logits(self) -> th.Tensor:
    """Returns the unnormalized log probabilities."""
    return self.dist.logits

  def sample(self, sample_shape: th.Size = th.Size()) -> th.Tensor:  # B008
    """Generates a sample_shape shaped sample or sample_shape shaped batch of
            samples if the distribution parameters are batched.

        Args:
            sample_shape (th.Size): The size of the sample to be drawn.

        Returns:
            A sample_shape shaped sample.
        """
    return self.dist.sample()

  def log_prob(self, actions: th.Tensor) -> th.Tensor:
    """Returns the log of the probability density/mass function evaluated at actions.

        Args:
            actions (th.Tensor): The actions to be evaluated.

        Returns:
            The log_prob value.
        """
    return self.dist.log_prob(actions)

  def entropy(self) -> th.Tensor:
    """Returns the Shannon entropy of distribution."""
    return self.dist.entropy()

  @property
  def mode(self) -> th.Tensor:
    """Returns the mode of the distribution."""
    return self.dist.probs.argmax(axis=-1)

  @property
  def mean(self) -> th.Tensor:
    """Returns the mean of the distribution."""
    return self.dist.probs.argmax(axis=-1)


class MultiCategorical(BaseDistribution):
  """Multi-categorical distribution for sampling actions for 'MultiDiscrete' tasks."""

  def __init__(self) -> None:
    super().__init__()

  def __call__(self, logits: Tuple[th.Tensor, ...]):
    """Create the distribution.

        Args:
            logits (Tuple[th.Tensor, ...]): The event log probabilities (unnormalized).

        Returns:
            Multi-categorical distribution instance.
        """
    super().__init__()
    self.dist = [pyd.Categorical(logits=logits_) for logits_ in logits]
    return self

  @property
  def probs(self) -> Tuple[th.Tensor, ...]:
    """Return probabilities."""
    return (dist.probs for dist in self.dist)  # type: ignore

  @property
  def logits(self) -> Tuple[th.Tensor, ...]:
    """Returns the unnormalized log probabilities."""
    return (dist.logits for dist in self.dist)  # type: ignore

  def sample(self, sample_shape: th.Size = th.Size()) -> th.Tensor:  # B008
    """Generates a sample_shape shaped sample or sample_shape shaped batch of
            samples if the distribution parameters are batched.

        Args:
            sample_shape (th.Size): The size of the sample to be drawn.

        Returns:
            A sample_shape shaped sample.
        """
    return th.stack([dist.sample() for dist in self.dist], dim=1)

  def log_prob(self, actions: th.Tensor) -> th.Tensor:
    """Returns the log of the probability density/mass function evaluated at actions.

        Args:
            actions (th.Tensor): The actions to be evaluated.

        Returns:
            The log_prob value.
        """
    return th.stack([
        dist.log_prob(action)
        for dist, action in zip(self.dist, th.unbind(actions, dim=1))
    ],
                    dim=1).sum(dim=1)

  def entropy(self) -> th.Tensor:
    """Returns the Shannon entropy of distribution."""
    return th.stack([dist.entropy() for dist in self.dist], dim=1).sum(dim=1)

  @property
  def mode(self) -> th.Tensor:
    """Returns the mode of the distribution."""
    return th.stack([dist.probs.argmax(axis=-1) for dist in self.dist], dim=1)

  @property
  def mean(self) -> th.Tensor:
    """Returns the mean of the distribution."""
    return th.stack([dist.probs.argmax(axis=-1) for dist in self.dist], dim=1)


class TanhTransform(pyd.transforms.Transform):
  # Borrowed from https://github.com/denisyarats/pytorch_sac/blob/master/agent/actor.py.
  """Tanh transformation."""

  domain = pyd.constraints.real
  codomain = pyd.constraints.interval(-1.0, 1.0)
  bijective = True
  sign = +1

  def __init__(self, cache_size=1):
    super().__init__(cache_size=cache_size)

  @staticmethod
  def atanh(x):
    return 0.5 * (x.log1p() - (-x).log1p())

  def __eq__(self, other):
    return isinstance(other, TanhTransform)

  def _call(self, x):
    return x.tanh()

  def _inverse(self, y):
    return self.atanh(y)

  def log_abs_det_jacobian(self, x, y):
    return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class SquashedNormal(BaseDistribution):
  """Squashed normal distribution for `Box` tasks."""

  def __init__(self) -> None:
    super().__init__()

  def __call__(self, mu: th.Tensor, sigma: th.Tensor):
    """Create the distribution.

        Args:
            mu (th.Tensor): The mean of the distribution.
            sigma (th.Tensor): The standard deviation of the distribution.

        Returns:
            Squashed normal distribution instance.
        """
    self.mu = mu
    self.sigma = sigma
    self.dist = pyd.TransformedDistribution(
        base_distribution=pyd.Normal(loc=mu, scale=sigma),
        transforms=[TanhTransform()],
    )
    return self

  def sample(self, sample_shape: th.Size = th.Size()) -> th.Tensor:  # B008
    """Generates a sample_shape shaped sample or sample_shape shaped
            batch of samples if the distribution parameters are batched.

        Args:
            sample_shape (th.Size): The size of the sample to be drawn.

        Returns:
            A sample_shape shaped sample.
        """
    return self.dist.sample(sample_shape)

  def rsample(self, sample_shape: th.Size = th.Size()) -> th.Tensor:  # B008
    """Generates a sample_shape shaped reparameterized sample or sample_shape shaped
            batch of reparameterized samples if the distribution parameters are batched.

        Args:
            sample_shape (th.Size): The size of the sample to be drawn.

        Returns:
            A sample_shape shaped sample.
        """
    return self.dist.rsample(sample_shape)

  @property
  def mean(self) -> th.Tensor:
    """Return the transformed mean."""
    loc = self.mu
    for tr in self.dist.transforms:
      loc = tr(loc)
    return loc

  @property
  def mode(self) -> th.Tensor:
    """Returns the mode of the distribution."""
    return self.mean

  def log_prob(self, actions: th.Tensor) -> th.Tensor:
    """Scores the sample by inverting the transform(s) and computing the score using
            the score of the base distribution and the log abs det jacobian.
        Args:
            actions (th.Tensor): The actions to be evaluated.

        Returns:
            The log_prob value.
        """
    return self.dist.log_prob(actions)


SelfDiagonalGaussian = TypeVar("SelfDiagonalGaussian", bound="DiagonalGaussian")


class DiagonalGaussian(BaseDistribution):
  """Diagonal Gaussian distribution for 'Box' tasks."""

  def __init__(self) -> None:
    super().__init__()

  def __call__(
      self: SelfDiagonalGaussian, mu: th.Tensor, sigma: th.Tensor
  ) -> SelfDiagonalGaussian:
    """Create the distribution.

        Args:
            mu (th.Tensor): The mean of the distribution.
            sigma (th.Tensor): The standard deviation of the distribution.

        Returns:
            Diagonal Gaussian distribution instance.
        """
    self.mu = mu
    self.sigma = sigma
    self.dist = pyd.Normal(loc=mu, scale=sigma)
    return self

  def sample(self, sample_shape: th.Size = th.Size()) -> th.Tensor:  # B008
    """Generates a sample_shape shaped sample or sample_shape shaped batch of
            samples if the distribution parameters are batched.

        Args:
            sample_shape (th.Size): The size of the sample to be drawn.

        Returns:
            A sample_shape shaped sample.
        """
    return self.dist.sample(sample_shape)

  def rsample(self, sample_shape: th.Size = th.Size()) -> th.Tensor:  # B008
    """Generates a sample_shape shaped reparameterized sample or sample_shape shaped batch of
            reparameterized samples if the distribution parameters are batched.

        Args:
            sample_shape (th.Size): The size of the sample to be drawn.

        Returns:
            A sample_shape shaped sample.
        """
    return self.dist.rsample(sample_shape)

  @property
  def mean(self) -> th.Tensor:
    """Returns the mean of the distribution."""
    return self.mu

  @property
  def mode(self) -> th.Tensor:
    """Returns the mode of the distribution."""
    return self.mu

  @property
  def stddev(self) -> th.Tensor:
    """Returns the standard deviation of the distribution."""
    raise self.dist.scale

  @property
  def variance(self) -> th.Tensor:
    """Returns the variance of the distribution."""
    return self.stddev.pow(2)

  def log_prob(self, actions: th.Tensor) -> th.Tensor:
    """Returns the log of the probability density/mass function evaluated at actions.

        Args:
            actions (th.Tensor): The actions to be evaluated.

        Returns:
            The log_prob value.
        """
    return self.dist.log_prob(actions).sum(-1)

  def entropy(self) -> th.Tensor:
    """Returns the Shannon entropy of distribution."""
    return self.dist.entropy()


@register_kl(Bernoulli, Bernoulli)
def kl_bernoulli_bernoulli(p, q):
  t1 = p.probs * (
      th.nn.functional.softplus(-q.logits) -
      th.nn.functional.softplus(-p.logits)
  )
  t1[q.probs == 0] = th.inf
  t1[p.probs == 0] = 0
  t2 = (1 - p.probs) * (
      th.nn.functional.softplus(q.logits) - th.nn.functional.softplus(p.logits)
  )
  t2[q.probs == 1] = th.inf
  t2[p.probs == 1] = 0
  return t1 + t2


@register_kl(Categorical, Categorical)
def kl_categorical_categorical(p, q):
  t = p.probs * (p.logits - q.logits)
  t[(q.probs == 0).expand_as(t)] = th.inf
  t[(p.probs == 0).expand_as(t)] = 0
  return t.sum(-1)


@register_kl(DiagonalGaussian, DiagonalGaussian)
def kl_diagonal_gaussian_diagonal_gaussian(p, q):
  var_ratio = (p.scale / q.scale).pow(2)
  t1 = ((p.loc - q.loc) / q.scale).pow(2)
  return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())


class NormalNoise(BaseDistribution):
  """Gaussian action noise.

    Args:
        mu (Union[float, th.Tensor]): Mean of the noise.
        sigma (Union[float, th.Tensor]): Standard deviation of the noise.
        low (float): The lower bound of the noise.
        high (float): The upper bound of the noise.
        eps (float): A small value to avoid numerical instability.

    Returns:
        Gaussian action noise instance.
    """

  def __init__(
      self,
      mu: Union[float, th.Tensor] = 0.0,
      sigma: Union[float, th.Tensor] = 1.0,
      low: float = -1.0,
      high: float = 1.0,
      eps: float = 1e-6,
  ) -> None:
    super().__init__()

    self.mu = mu
    self.sigma = sigma
    self.low = low
    self.high = high
    self.eps = eps
    self.dist = pyd.Normal(loc=mu, scale=sigma)

  def __call__(self, noiseless_action: th.Tensor):
    """Create the action noise.

        Args:
            noiseless_action (th.Tensor): Unprocessed actions.

        Returns:
            Normal noise instance.
        """
    self.noiseless_action = noiseless_action
    return self

  def _clamp(self, x: th.Tensor) -> th.Tensor:
    """Clamps the input to the range [low, high]."""
    clamped_x = th.clamp(x, self.low + self.eps, self.high - self.eps)
    x = x - x.detach() + clamped_x.detach()
    return x

  def sample(
      self,
      clip: Optional[float] = None,
      sample_shape: th.Size = th.Size()
  ) -> th.Tensor:  # type: ignore[override]
    """Generates a sample_shape shaped sample or sample_shape shaped batch of
            samples if the distribution parameters are batched.

        Args:
            clip (Optional[float]): The clip range of the sampled noises.
            sample_shape (th.Size): The size of the sample to be drawn.

        Returns:
            A sample_shape shaped sample.
        """
    noise = th.as_tensor(
        self.dist.sample(sample_shape=self.noiseless_action.size()),
        device=self.noiseless_action.device,
        dtype=self.noiseless_action.dtype,
    )

    if clip is not None:
      # clip the sampled noises
      noise = th.clamp(noise, -clip, clip)
    return self._clamp(noise + self.noiseless_action)

  @property
  def mean(self) -> th.Tensor:
    """Returns the mean of the distribution."""
    return self.noiseless_action

  @property
  def mode(self) -> th.Tensor:
    """Returns the mode of the distribution."""
    return self.noiseless_action


class OrnsteinUhlenbeckNoise(BaseDistribution):
  """Ornstein Uhlenbeck action noise.
        Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab

    Args:
        mu (Union[float, th.Tensor]): Mean of the noise.
        sigma (Union[float, th.Tensor]): Standard deviation of the noise.
        low (float): The lower bound of the noise.
        high (float): The upper bound of the noise.
        eps (float): A small value to avoid numerical instability.
        theta (float): The rate of mean reversion.
        dt (float): Timestep for the noise.
        stddev_schedule (str): Use the exploration std schedule.
        stddev_clip (float): The exploration std clip range.

    Returns:
        Ornstein-Uhlenbeck noise instance.
    """

  def __init__(
      self,
      mu: Union[float, th.Tensor] = 0.0,
      sigma: Union[float, th.Tensor] = 1.0,
      low: float = -1.0,
      high: float = 1.0,
      eps: float = 1e-6,
      theta: float = 0.15,
      dt: float = 1e-2,
  ) -> None:
    super().__init__()

    self.mu = mu
    self.sigma = sigma
    self.low = low
    self.high = high
    self.eps = eps
    self.theta = theta
    self.dt = dt
    self.noise_prev: Union[None, th.Tensor] = None

  def __call__(self, noiseless_action: th.Tensor):
    """Create the action noise.

        Args:
            noiseless_action (th.Tensor): Unprocessed actions.

        Returns:
            Ornstein-Uhlenbeck noise instance.
        """
    self.noiseless_action = noiseless_action
    if self.noise_prev is None:
      self.noise_prev = th.zeros_like(self.noiseless_action)
    return self

  def _clamp(self, x: th.Tensor) -> th.Tensor:
    """Clamps the input to the range [low, high]."""
    clamped_x = th.clamp(x, self.low + self.eps, self.high - self.eps)
    x = x - x.detach() + clamped_x.detach()
    return x

  def sample(
      self,
      clip: Optional[float] = None,
      sample_shape: th.Size = th.Size()
  ) -> th.Tensor:  # type: ignore[override]
    """Generates a sample_shape shaped sample or sample_shape shaped batch of
            samples if the distribution parameters are batched.

        Args:
            clip (Optional[float]): The clip range of the sampled noises.
            sample_shape (th.Size): The size of the sample to be drawn.

        Returns:
            A sample_shape shaped sample.
        """
    noise = (
        self.noise_prev + self.theta *
        (th.ones_like(self.noise_prev) * self.mu - self.noise_prev) *
        self.dt  # type: ignore
        + self.sigma * (self.dt ** 0.5) * _standard_normal(
            self.noiseless_action.size(),
            dtype=self.noiseless_action.dtype,
            device=self.noiseless_action.device,
        )
    )
    noise = th.as_tensor(
        noise,
        dtype=self.noiseless_action.dtype,
        device=self.noiseless_action.device,
    )
    self.noise_prev = noise

    if clip is not None:
      # clip the sampled noises
      noise = th.clamp(noise, -clip, clip)

    return self._clamp(noise + self.noiseless_action)

  def reset(self) -> None:
    """Reset the noise."""
    self.noise_prev = None

  @property
  def mean(self) -> th.Tensor:
    """Returns the mean of the distribution."""
    return self.noiseless_action

  @property
  def mode(self) -> th.Tensor:
    """Returns the mode of the distribution."""
    return self.noiseless_action


def schedule(schdl: str, step: int) -> float:
  """Exploration noise schedule.

    Args:
        schdl (str): Schedule mode.
        step (int): global training step.

    Returns:
        Standard deviation.
    """
  try:
    return float(schdl)
  except ValueError:
    match = re.match(r"linear\((.+),(.+),(.+)\)", schdl)
    if match:
      init, final, duration = (float(g) for g in match.groups())
      mix = np.clip(step / duration, 0.0, 1.0)
      return (1.0 - mix) * init + mix * final
    match = re.match(r"step_linear\((.+),(.+),(.+),(.+),(.+)\)", schdl)
    if match:
      init, final1, duration1, final2, duration2 = (
          float(g) for g in match.groups()
      )
      if step <= duration1:
        mix = np.clip(step / duration1, 0.0, 1.0)
        return (1.0 - mix) * init + mix * final1
      else:
        mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
        return (1.0 - mix) * final1 + mix * final2
  raise NotImplementedError(schdl)


class TruncatedNormalNoise(BaseDistribution):
  """Truncated normal action noise. See Section 3.1 of
        "Mastering Visual Continuous Control: Improved Data-Augmented Reinforcement Learning".

    Args:
        mu (Union[float, th.Tensor]): Mean of the noise.
        sigma (Union[float, th.Tensor]): Standard deviation of the noise.
        low (float): The lower bound of the noise.
        high (float): The upper bound of the noise.
        eps (float): A small value to avoid numerical instability.
        stddev_schedule (str): Use the exploration std schedule, available options are:
            `linear(init, final, duration)` and `step_linear(init, final1, duration1, final2, duration2)`.

    Returns:
        Truncated normal noise instance.
    """

  def __init__(
      self,
      mu: Union[float, th.Tensor] = 0.0,
      sigma: Union[float, th.Tensor] = 1.0,
      low: float = -1.0,
      high: float = 1.0,
      eps: float = 1e-6,
      stddev_schedule: str = "linear(1.0, 0.1, 100000)",
  ) -> None:
    super().__init__()

    self.mu = mu
    self.sigma = sigma
    self.low = low
    self.high = high
    self.eps = eps
    self.stddev_schedule = stddev_schedule
    self.step = 0

  def __call__(self, noiseless_action: th.Tensor):
    """Create the action noise.

        Args:
            noiseless_action (th.Tensor): Unprocessed actions.

        Returns:
            Truncated normal noise instance.
        """
    self.noiseless_action = noiseless_action
    self.scale = schedule(self.stddev_schedule, self.step)
    return self

  def _clamp(self, x: th.Tensor) -> th.Tensor:
    """Clamps the input to the range [low, high]."""
    clamped_x = th.clamp(x, self.low + self.eps, self.high - self.eps)
    x = x - x.detach() + clamped_x.detach()
    return x

  def sample(
      self,
      clip: Optional[float] = None,
      sample_shape: th.Size = th.Size()
  ) -> th.Tensor:  # type: ignore[override]
    """Generates a sample_shape shaped sample or sample_shape shaped batch of
            samples if the distribution parameters are batched.

        Args:
            clip (Optional[float]): The clip range of the sampled noises.
            sample_shape (th.Size): The size of the sample to be drawn.

        Returns:
            A sample_shape shaped sample.
        """
    noise = _standard_normal(
        self.noiseless_action.size(),
        dtype=self.noiseless_action.dtype,
        device=self.noiseless_action.device
    )
    noise *= self.scale

    if clip is not None:
      # clip the sampled noises
      noise = th.clamp(noise, -clip, clip)

    self.step += 1

    return self._clamp(noise + self.noiseless_action)

  @property
  def mean(self) -> th.Tensor:
    """Returns the mean of the distribution."""
    return self.noiseless_action

  @property
  def mode(self) -> th.Tensor:
    """Returns the mode of the distribution."""
    return self.noiseless_action
