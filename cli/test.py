import argparse
from utils.atari_util import *
from utils.utils import *
from models.self_attn_cnn_gru import *
import torch
import yaml
import gym.wrappers
import os


def parse_args(
) -> argparse.Namespace:

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default=None, help="Path to the config.")

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.load(f)["params"]

    env = make_env(config["common"]["env_name"],
                   config["common"]["height"],
                   config["common"]["width"]
                  )

    if config["paths"]["trained_model_path_to_test"] is None:
        print("trained model's path is empty - check it!")
        config["paths"]["trained_model_path_to_test"] = "./"

    model_path: str = os.path.join(config["paths"]["trained_model_path_to_test"], config["experiment"] + ".pt")
    videos_path: str = config["paths"]["test_stupid_videos_path"]

    if os.path.isfile(model_path):
        agent = torch.load(model_path)
        videos_path: str = config["paths"]["test_trained_videos_path"]
    else:

        agent = SelfAttnRecurrentAgent(obs_shape=env.observation_space.shape,
                                       n_actions=env.action_space.n,
                                       linear_dim=config["model"]["linear_dim"],
                                       conv_filters_num=config["model"]["conv_filters_num"],
                                       hidden_dim=config["model"]["hidden_dim"]
                                      )
        if config["paths"]["trained_weigths_path"]:
            agent.load_state_dict(torch.load(config["paths"]["trained_weights_path"]))

    with gym.wrappers.Monitor(env=env,
                              directory=videos_path,
                              force=True) as env_monitor:
        final_rewards = evaluate(agent, env_monitor, n_games=config["test"]["game_count"])