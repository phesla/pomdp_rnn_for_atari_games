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


# функция для парсинга входных параметров программы: передаем путь до yml-конфига с нужными нам параметрами.
def parse_args(
) -> argparse.Namespace:

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default=None, help="Path to the config.")

    return parser.parse_args()

# функция для тренировки модели на некотором временном ряду данных среды
# (rollout: в нашем случае 10 тактов игры): состояния среды (картинки, описывающие некоторый промежуток времени в игре)
def train_on_rollout(
        states,
        actions,
        rewards,
        prev_memory_states,
        n_actions: int,
        gamma: float = 0.99
):
    """
    Takes a sequence of states, actions and rewards produced by generate_session.
    Updates agent's weights by following the policy gradient above.
    Please use Adam optimizer with default parameters.
    """

    # тензор состояний среды
    states = torch.tensor(np.asarray(states), dtype=torch.float32)
    # тензор действий среды
    actions = torch.tensor(np.array(actions), dtype=torch.int64)  # shape: [batch_size, time]
    # тензор наград модели
    rewards = torch.tensor(np.array(rewards), dtype=torch.float32)  # shape: [batch_size, time]
    rollout_length = rewards.shape[1] - 1

    # отключаем автоматический подсчет градиента по тензору памяти
    # (предыдущее скрытое состояние для рекуррентного слоя GRU, вектор)
    memory = prev_memory_states.detach()

    # будем собирать логиты (логарифмы от) предсказаний модели, которая предсказывает вектор вероятностей
    # (вероятности для каждого из всех возможных действий), а также собираем
    logits = []
    state_values = []

    # итерируемся по всем моментам времени (в нашем текущем случае их 10)
    for t in range(rewards.shape[1]):

        # берем t-ое состояние среды (картинку с номером t)
        obs_t = states[:, t]

        # передаем текущие входные данные в модель и делаем предсказание, а также возвращаем
        # новое скрытое состояние слоя GRU
        memory, (logits_t, values_t) = agent.forward(memory, obs_t)

        # сохраняем предсказания модели
        logits.append(logits_t)
        state_values.append(values_t)

    # соединяем тензоры в один вдоль 1-й размерности, применяем функции активации softmax и
    # логарифм от softmax, чтобы получить вероятности и натуральные логарифмы этих же вероятностей
    logits = torch.stack(logits, dim=1)
    state_values = torch.stack(state_values, dim=1)
    probas = F.softmax(logits, dim=2)
    logprobas = F.log_softmax(logits, dim=2)

    # считаем логарифмы вероятностей для всех возможных действий агента с помощью предсказаний
    # модели, затем суммируем их по времени
    actions_one_hot = to_one_hot(actions, n_actions).view(
        actions.shape[0], actions.shape[1], n_actions)
    logprobas_for_actions = torch.sum(logprobas * actions_one_hot, dim=-1)

    # функции ошибки (подробно описаны в документации)
    J_hat = 0
    value_loss = 0

    # G(t) - кумулятивная награда агента
    cumulative_returns = state_values[:, -1].detach()

    # итерируемся по всем тактам промежутка времени
    for t in reversed(range(rollout_length)):
        r_t = rewards[:, t]
        # текущее значение функции V(s) на шаге t (подробно описана в документации)
        V_t = state_values[:, t]
        V_next = state_values[:, t + 1].detach()
        # логарифм вероятности a(t) при условии s(t)
        logpi_a_s_t = logprobas_for_actions[:, t]

        # обновляем G_t = r_t + gamma * G_{t+1}
        cumulative_returns = G_t = r_t + gamma * cumulative_returns

        # считаем MSE для сходимости V(s) к |E (R(s, a)) при a~p(a|s)
        value_loss += torch.mean((r_t + gamma * V_next - V_t) ** 2)

        # считаем пользу A(s_t, a_t), используя кумулятивную функцию значений V(s(t)) как базис
        advantage = cumulative_returns - V_t
        advantage = advantage.detach()

        # считаем функцию политики (подробно описана в документации)
        J_hat += torch.mean(logpi_a_s_t * advantage)

    # считаем функцию энтропии (описана в документации)
    entropy_reg = -torch.mean(torch.sum(probas * logprobas, dim=-1))

    # собираем все функции ошибки с коэффициентами в одну функцию
    loss = -J_hat / rollout_length + value_loss / rollout_length - 0.01 * entropy_reg

    # зануляем векторы градиентов, считаем градиенты и применяем их к параметрам модели
    opt.zero_grad()
    loss.backward()
    opt.step()

    # возвращаем numpy-массив функции ошибки
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
    # список с историей наград и фигура для отрисовки графика изменения наград
    rewards_history: List[float] = []
    plt_fig = plt.figure()

    # процесс обучения
    for i in trange(iter_count):

        # берем данные за 10 последовательных тактов 5 параллельных игр
        memory = pool.prev_memory_states
        rollout_obs, rollout_actions, rollout_rewards, rollout_mask = pool.interact(10)

        # учим модель на текущих полученных данных
        train_on_rollout(rollout_obs, rollout_actions, rollout_rewards, memory, n_actions)

        # каждые 100 итераций проверяем качество модели в реальной игре, считая награду
        # затем строим отмечаем точку на графике: по оси x - время, по оси y - награда модели
        if i % 100 == 0:
            rewards_history.append(np.mean(evaluate(agent, env, n_games=1)))

            # когда модель достигает заранее определенного нами значения награды в ходе
            # моделирования реальной игры, сохраняем модель, сохраняем график и останавливаем обучение
            if rewards_history[-1] >= reward_limit:
                print("Your agent has just passed the reward limit.")
                plt.plot(rewards_history, label='rewards')
                plt.legend()
                torch.save(agent, weigths_path)
                plt_fig.savefig(reward_figure_path)
                break


if __name__ == "__main__":

    # парсим аргументы
    args = parse_args()

    # читаем данные из конфига - получаем словарь
    with open(args.config_path, "r") as f:
        config = yaml.load(f)["params"]

    # объявляем переменные для путей для сохранения весов модели и для сохранения
    # графика наград при тестировании
    path_to_save_weights: str = os.path.join(config["paths"]["weigths_path_to_save"],
                                             config["experiment"] + ".pt")
    path_to_save_figure: str = os.path.join(config["paths"]["figures_path"],
                                            config["experiment"] + ".png")

    # создаем среду gym для моделирования игрового процесса
    env = make_env(config["common"]["env_name"],
                   config["common"]["height"],
                   config["common"]["width"]
                  )

    # загружаем сохраненную обученную модель, либо инициализируем новую модель с обученными весами
    # или без них
    if config["paths"]["trained_model_path_to_train"]:
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

    # инициализируем пул сред для параллельного моделирования нескольких игр
    pool = EnvPool(agent,
                   make_env,
                   config["common"]["env_name"],
                   config["common"]["height"],
                   config["common"]["width"],
                   config["train"]["parallels_game_count"])

    # инициализируем метод оптимизации нашей модели: Adam чаще всего лучше показывает себя, чем
    # классический стохастический градиентный спуск (пусть даже и с моментом Нестерова), так как
    # учитывает ограниченное множество обновлений каждого параметра модели (менее яркие признаки данных
    # слабее влияют на параметры модели, которые за них отвечают), а также учитывает накопление движения (момент).
    opt = torch.optim.Adam(agent.parameters(), lr=config["train"]["lr"])

    # обучаем модель
    train(env=env,
          agent=agent,
          pool=pool,
          iter_count=config["train"]["iter_count"],
          reward_limit=config["train"]["reward_limit"],
          weigths_path=path_to_save_weights,
          reward_figure_path=path_to_save_figure,
          n_actions=env.action_space.n
         )