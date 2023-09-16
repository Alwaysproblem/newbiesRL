"""The main entry point of the program."""
import argparse
import json

from AC.main import main as ac_main
from PPO.main import main as ppo_main
from DQN.main import main as dqn_main
from DDQN.main import main as ddqn_main
from DDPG.main import main as ddpg_main
from C51.main import main as c51_main
from AWR.main import main as awr_main


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", help="config file path")
  args = parser.parse_args()
  if args.config is None:
    raise ValueError("config file path is not specified")
  config = args.config
  with open(config, "r", encoding="utf-8") as f:
    config = json.load(f)
  algo_table = {
      "AC": ac_main,
      "PPO": ppo_main,
      "DQN": dqn_main,
      "DDQN": ddqn_main,
      "DDPG": ddpg_main,
      "C51": c51_main,
      "AWR": awr_main
  }
  agent_main = algo_table.get(config["agent"], None)
  if not callable(agent_main):
    raise NotImplementedError(
        f"agent {config['agent']} is not implemented yet, "
        f"please choose from {list(algo_table.keys())}"
    )
  agent_main(**config)


if __name__ == "__main__":
  main()
