import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class SelfAttnRecurrentAgent(nn.Module):

    def __init__(
            self,
            obs_shape,
            n_actions,
            linear_dim: int,
            conv_filters_num: int = 32,
            hidden_dim: int = 128,
            dropout: float = 0.2,
            bidirectional: bool = False,
            batch_first: bool = True,
            num_gru_layers: int = 2
    ):
        """A actor-critic agent"""
        super(self.__class__, self).__init__()

        self.conv_filters_num = conv_filters_num
        self.linear_dim = linear_dim
        self.in_linear_size = 0
        self.hidden_dim = hidden_dim
        self.num_gru_layers = num_gru_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.dropout = dropout

        self.conv0 = nn.Conv2d(1, self.conv_filters_num, kernel_size=(3, 3), stride=(2, 2))
        self.conv1 = nn.Conv2d(self.conv_filters_num, self.conv_filters_num, kernel_size=(3, 3), stride=(2, 2))
        self.conv2 = nn.Conv2d(self.conv_filters_num, self.conv_filters_num, kernel_size=(3, 3), stride=(2, 2))
        self.flatten = nn.Flatten()

        self.hid = nn.Linear(self.conv_filters_num * 4 * 4, self.linear_dim)
        self.gru = nn.GRUCell(self.linear_dim, self.hidden_dim)
        self.attn = nn.Linear(self.linear_dim, self.hidden_dim)
        self.activation = nn.ELU()

        self.logits = nn.Linear(self.hidden_dim, n_actions)
        self.state_value = nn.Linear(self.hidden_dim, 1)

    def forward(
            self,
            hidden_prev_state,
            obs_t
    ):
        """
        Takes agent's previous hidden state and a new observation,
        returns a new hidden state and whatever the agent needs to learn
        """

        h = self.conv0(obs_t)
        h = self.activation(h)
        h = self.conv1(h)
        h = self.activation(h)
        h = self.conv2(h)
        h = self.activation(h)

        flatten = self.flatten(h)
        h = self.activation(self.hid(flatten))

        attn_map = F.softmax(self.attn(h), dim=1)
        attn_h = hidden_prev_state * attn_map
        new_hidden_state = self.gru(h, attn_h)

        logits = self.logits(new_hidden_state)
        state_value = self.state_value(new_hidden_state)

        return new_hidden_state, (logits, state_value)

    def get_initial_state(
            self,
            batch_size: int
    ):
        """Return a list of agent memory states at game start. Each state is a np array of shape [batch_size, ...]"""
        return torch.zeros((batch_size, self.hidden_dim))

    def sample_actions(
            self,
            agent_outputs
    ):
        """pick actions given numeric agent outputs (np arrays)"""
        logits, state_values = agent_outputs
        probs = F.softmax(logits)
        return torch.multinomial(probs, 1)[:, 0].data.numpy()

    def step(
            self,
            prev_state,
            obs_t
    ):
        """ like forward, but obs_t is a numpy array """
        obs_t = torch.tensor(np.asarray(obs_t), dtype=torch.float32)
        h, (l, s) = self.forward(prev_state, obs_t)
        return (h.detach()), (l.detach(), s.detach())