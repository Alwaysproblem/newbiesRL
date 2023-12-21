# Naive Implementations of Deep Reinforcement Learning

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

## About

This is some naive implementations of deep reinforcement learning algorithms. The purpose of this repo is to help me understand the algorithms and the code. The code is not optimized for performance. If you want to use the code for your research, please refer to the original paper and the official implementation. I verify the code with [OpenAI gymnasium](https://github.com/Farama-Foundation/Gymnasiu). The most of games that I used is `LunarLander-v2`, `CartPole-v1` and `Pendulum-v1`

<!-- ## Enviornment Preparation

- common lib

  ```bash
  conda install jupyter pandas colorama pylint yapf seaborn scipy scikit-learn tqdm tensorboardx==2.5.* tensorboard==2.* pillow -y -c conda-forge
  ```

- gymnasium
  - install gymnasium

    ```bash
    conda install gymnasium pyglet pygame gymnasium-box2d -c conda-forge
    ```

- gymnasium + pytorch

  ```bash
  conda install pytorch torchvision torchaudio -c pytorch
  ```

- gymnasium + TF2

  ```bash
  conda install tensorflow -c conda-forge
  ```

- gymnasium + jax

  ```bash
  conda install jax chex optax dm-haiku jaxlib Jraph -c conda-forge
  pip install coax
  ``` -->

## Table of Contents

- [Naive Implementations of Deep Reinforcement Learning](#naive-implementations-of-deep-reinforcement-learning)
  - [About](#about)
  - [Table of Contents](#table-of-contents)
  - [Environment Preparation (torch users)](#environment-preparation-torch-users)
  - [Run](#run)
  - [Algorithms](#algorithms)
  - [Reference](#reference)

## Environment Preparation (torch users)

```bash
conda create -n rltorch pytorch torchvision torchaudio pytorch-cuda=12.1 gymnasium pyglet pygame gymnasium-box2d colorama pylint yapf tqdm 'tensorboardx>=2.5.0' 'tensorboard>2.0' pillow matplotlib scipy seaborn ipykernel -c conda-forge -c pytorch -c nvidia
```

## Run

This project does not provide the trained Deep Reinforcement Learning model weight.

You can start training model under conda environment by

```bash
(rltorch) > python -m <project name>.main
```

For example (DDPG):

```bash
(rltorch) > python -m DDPG.main
```

## Algorithms

- [x] [DQN](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

  ![DQN](DQN/DQNAgent_200.gif)

- [x] [DDQN](https://arxiv.org/pdf/1509.06461.pdf)

  ![DDQN](DDQN/DDQNAgent_100.gif)

- [x] [DDPG](https://arxiv.org/pdf/1509.02971.pdf)

  ![DDPG](DDPG/DDPGAgent_200.gif)

- [x] [PPO](https://arxiv.org/pdf/1707.06347.pdf)

  ![PPO](PPO/PPOAgent_200.gif)

- [x] [Distributed Q learning (C51)](https://arxiv.org/pdf/1707.06887.pdf)

  ![C51](C51/C51Agent_100.gif)

- [x] [AWR](https://openreview.net/attachment?id=H1gdF34FvS&name=original_pdf)

  ![AWR](AWR/AWRAgent_200.gif)

- [x] [AC](https://proceedings.neurips.cc/paper/1999/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)

  ![AC](AC/A2CAgent_600.gif)

- improve `AWR`, `DDPG` with Gumbel Distribution Regression from [`XQL`](https://div99.github.io/XQL):
  - XAWR

    ![XAWR](XAWR/XAWRAgent_100.gif)

  - XDDPG

    ![XDDPG](XDDPG/XDDPGAgent_200.gif)

## Reference

- TrainMonitor and Generategif modified from [coax](https://github.com/coax-dev/coax)
- https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
- https://arxiv.org/pdf/1509.06461.pdf
- https://arxiv.org/pdf/1509.02971.pdf
- https://arxiv.org/pdf/1707.06347.pdf
- https://arxiv.org/pdf/1707.06887.pdf
- https://openreview.net/attachment?id=H1gdF34FvS&name=original_pdf
- https://proceedings.neurips.cc/paper/1999/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf
- https://div99.github.io/XQL/
