import os
import argparse
from utils.utils import *
from utils.atari_util import *
from models.self_attn_cnn_gru import *
import numpy as np
from tqdm import trange
import torch
from torch.nn import functional as F
import yaml
from typing import List, Dict
from utils.env_pool import EnvPool
import matplotlib.pyplot as plt


def parse_args(
) -> argparse.Namespace:

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default=None, help="Path to the config.")

    return parser.parse_args()


def train_on_rollout(
        states,
        actions,
        rewards,
        is_not_done,
        prev_memory_states,
        n_actions: int,
        gamma: float = 0.99
):
    """
    Takes a sequence of states, actions and rewards produced by generate_session.
    Updates agent's weights by following the policy gradient above.
    Please use Adam optimizer with default parameters.
    """

    # shape: [batch_size, time, c, h, w]
    states = torch.tensor(np.asarray(states), dtype=torch.float32)
    actions = torch.tensor(np.array(actions), dtype=torch.int64)  # shape: [batch_size, time]
    rewards = torch.tensor(np.array(rewards), dtype=torch.float32)  # shape: [batch_size, time]
    is_not_done = torch.tensor(np.array(is_not_done), dtype=torch.float32)  # shape: [batch_size, time]
    rollout_length = rewards.shape[1] - 1

    memory = prev_memory_states.detach()

    logits = []
    state_values = []
    for t in range(rewards.shape[1]):
        obs_t = states[:, t]
        memory, (logits_t, values_t) = agent.forward(memory, obs_t)

        logits.append(logits_t)
        state_values.append(values_t)

    logits = torch.stack(logits, dim=1)
    state_values = torch.stack(state_values, dim=1)
    probas = F.softmax(logits, dim=2)
    logprobas = F.log_softmax(logits, dim=2)

    actions_one_hot = to_one_hot(actions, n_actions).view(
        actions.shape[0], actions.shape[1], n_actions)
    logprobas_for_actions = torch.sum(logprobas * actions_one_hot, dim=-1)

    J_hat = 0
    value_loss = 0

    cumulative_returns = state_values[:, -1].detach()

    for t in reversed(range(rollout_length)):
        r_t = rewards[:, t]                                # current rewards
        # current state values
        V_t = state_values[:, t]
        V_next = state_values[:, t + 1].detach()           # next state values
        # log-probability of a_t in s_t
        logpi_a_s_t = logprobas_for_actions[:, t]

        # update G_t = r_t + gamma * G_{t+1} as we did in week6 reinforce
        cumulative_returns = G_t = r_t + gamma * cumulative_returns

        # Compute temporal difference error (MSE for V(s))
        value_loss += torch.mean((r_t + gamma * V_next - V_t) ** 2)

        # compute advantage A(s_t, a_t) using cumulative returns and V(s_t) as baseline
        advantage = cumulative_returns - V_t
        advantage = advantage.detach()

        # compute policy pseudo-loss aka -J_hat.
        J_hat += torch.mean(logpi_a_s_t * advantage)

    # regularize with entropy
    entropy_reg = -torch.mean(torch.sum(probas * logprobas, dim=-1))

    # add-up three loss components and average over time
    loss = -J_hat / rollout_length +\
        value_loss / rollout_length +\
           -0.01 * entropy_reg

    # Gradient descent step
    opt.zero_grad()
    loss.backward()
    opt.step()

    return loss.data.numpy()

def train(
        env,
        agent,
        pool,
        iter_count: int,
        reward_limit: float,
        weigths_path: str,
        reward_figure_path: str,
        n_actions: int
):
    rewards_history: List[float] = []
    plt_fig = plt.figure()

    for i in trange(iter_count):

        memory = pool.prev_memory_states
        rollout_obs, rollout_actions, rollout_rewards, rollout_mask = pool.interact(10)
        train_on_rollout(rollout_obs, rollout_actions,
                         rollout_rewards, rollout_mask, memory, n_actions)

        if i % 100 == 0:
            rewards_history.append(np.mean(evaluate(agent, env, n_games=1)))
            plt.plot(rewards_history, label='rewards')
            plt.legend()

            if rewards_history[-1] >= reward_limit:
                print("Your agent has just passed the reward limit.")
                torch.save(agent, weigths_path)
                plt_fig.savefig(reward_figure_path)
                break


if __name__ == "__main__":

    args = parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.load(f)["params"]

    path_to_save_weights: str = os.path.join(config["paths"]["weigths_path_to_save"],
                                             config["experiment"] + ".pt")
    path_to_save_figure: str = os.path.join(config["paths"]["figures_path"],
                                            config["experiment"] + ".png")

    env = make_env(config["common"]["env_name"],
                   config["common"]["height"],
                   config["common"]["width"]
                  )

    if config["paths"]["trained_model_path"]:
        agent = torch.load(config["paths"]["trained_model_path"])
    else:

        agent = SelfAttnRecurrentAgent(obs_shape=env.observation_space.shape,
                                       n_actions=env.action_space.n,
                                       linear_dim=config["model"]["linear_dim"],
                                       conv_filters_num=config["model"]["conv_filters_num"],
                                       hidden_dim=config["model"]["hidden_dim"]
                                      )
        if config["paths"]["trained_weigths_path"]:
            agent.load_state_dict(torch.load(config["paths"]["trained_weights_path"]))

    pool = EnvPool(agent,
                   make_env,
                   config["common"]["env_name"],
                   config["common"]["height"],
                   config["common"]["width"],
                   config["train"]["parallels_game_count"])

    opt = torch.optim.Adam(agent.parameters(), lr=config["train"]["lr"])

    train(env=env,
          agent=agent,
          pool=pool,
          iter_count=config["train"]["iter_count"],
          reward_limit=config["train"]["reward_limit"],
          weigths_path=path_to_save_weights,
          reward_figure_path=path_to_save_figure,
          n_actions=env.action_space.n
         )