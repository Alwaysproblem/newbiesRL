# RL-try
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

This is a practice for Deep reinforcement learning
## Enviornment Preparation

- common lib

  ```bash
  conda install jupyter pandas colorama pylint yapf seaborn scipy scikit-learn tqdm tensorboardx>=2.5 tensorboard pillow -y -c conda-forge
  ```

- gym
  - install gym

    ```bash
    conda install gym pyglet pygame -c conda-forge
    ```

- gym + pytorch

  ```bash
  conda install pytorch torchvision torchaudio -c pytorch
  ```

- gym + TF2

  ```bash
  conda install tensorflow -c conda-forge
  ```

- gym + jax

  ```bash
  conda install jax chex optax dm-haiku jaxlib Jraph -c conda-forge
  pip install coax
  ```


## Reference

- TrainMonitor and Generategif modified from [coax](https://github.com/coax-dev/coax)
