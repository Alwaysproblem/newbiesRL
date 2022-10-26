import os
import logging

from PIL import Image


def generate_gif(
    env,
    filepath,
    policy=None,
    resize_to=None,
    duration=50,
    max_episode_steps=None
):
  r"""
    Store a gif from the episode frames.
    Parameters
    ----------
    env : gym environment
        The environment to record from.
    filepath : str
        Location of the output gif file.
    policy : callable, optional
        A policy objects that is used to pick actions: ``a = policy(s)``. If left unspecified, we'll
        just take random actions instead, i.e. ``a = env.action_space.sample()``.
    resize_to : tuple of ints, optional
        The size of the output frames, ``(width, height)``. Notice the
        ordering: first **width**, then **height**. This is the convention PIL
        uses.
    duration : float, optional
        Time between frames in the animated gif, in milliseconds.
    max_episode_steps : int, optional
        The maximum number of step in the episode. If left unspecified, we'll
        attempt to get the value from ``env.spec.max_episode_steps`` and if
        that fails we default to 10000.
    """
  logger = logging.getLogger('generate_gif')
  max_episode_steps = max_episode_steps \
      or getattr(getattr(env, 'spec'), 'max_episode_steps', 10000)

  from .wrappers import TrainMonitor
  if isinstance(env, TrainMonitor):
    env = env.env  # unwrap to strip off TrainMonitor

  # collect frames
  frames = []
  s = env.reset()
  for t in range(max_episode_steps):
    a = env.action_space.sample() if policy is None else policy(s)
    s_next, r, done, info = env.step(a)

    # store frame
    frame = env.render(mode='rgb_array')
    frame = Image.fromarray(frame)
    frame = frame.convert('P', palette=Image.ADAPTIVE)
    if resize_to is not None:
      if not (isinstance(resize_to, tuple) and len(resize_to) == 2):
        raise TypeError("expected a tuple of size 2, resize_to=(w, h)")
      frame = frame.resize(resize_to)

    frames.append(frame)

    if done:
      break

    s = s_next

  # store last frame
  frame = env.render(mode='rgb_array')
  frame = Image.fromarray(frame)
  frame = frame.convert('P', palette=Image.ADAPTIVE)
  if resize_to is not None:
    frame = frame.resize(resize_to)
  frames.append(frame)

  # generate gif
  os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
  frames[0].save(
      fp=filepath,
      format='GIF',
      append_images=frames[1:],
      save_all=True,
      duration=duration,
      loop=0
  )

  logger.info("recorded episode to: {}".format(filepath))
