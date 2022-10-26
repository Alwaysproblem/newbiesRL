# RL-try
this is a practice for Deep reinforcement learning

## Enviornment Preparation

- common lib

  ```bash
  conda install jupyter pandas colorama pylint yapf seaborn scipy scikit-learn tqdm tensorboardx tensorboard pillow -y -c conda-forge
  ```

- gym
  - install gym

    ```bash
    conda install gym pyglet -c conda-forge
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
