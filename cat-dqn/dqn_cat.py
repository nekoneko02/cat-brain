import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.modules import DistributionalQValueActor

import adapter
importlib.reload(adapter)


class DuelingCategoricalHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_actions, num_atoms):
        super().__init__()
        self.num_actions = num_actions
        self.num_atoms = num_atoms

        self.value_stream = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_atoms)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions * num_atoms)
        )

    def forward(self, x):
        batch_size = x.size(0)

        value = self.value_stream(x)  # [B, num_atoms]
        advantage = self.advantage_stream(x).view(batch_size, self.num_actions, self.num_atoms)  # [B, A, N]

        value = value.unsqueeze(1)  # [B, 1, N]
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)  # [B, A, N]

        return q_atoms


class DQNCat(nn.Module):
    def __init__(self, dqn_config):
        super().__init__()
        input_dim = dqn_config["input_dim"]
        hidden_dim = dqn_config["rnn"]["hidden_dim"]
        self.num_actions = dqn_config["num_actions"]
        self.num_atoms = dqn_config["categorical"]["num_atoms"]
        self.vmin = dqn_config["categorical"]["v_min"]
        self.vmax = dqn_config["categorical"]["v_max"]

        # Adapter定義
        self.input_adapters = nn.ModuleList([
            adapter.RnnInputAdapter(dqn_config["rnn"]),
            adapter.InputAdapter(dqn_config)
        ])

        # ストリーム構築
        streams = []

        # RNN
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)

        feature_net = []
        for layer in dqn_config["feature_layers"]:
            feature_net.append(nn.LazyLinear(layer))
            feature_net.append(nn.ReLU())
        self.feature_net = nn.Sequential(*feature_net)

        medium_dim = dqn_config["feature_layers"][-1]

        # Dueling categorical head
        self.head = DuelingCategoricalHead(
            in_dim=medium_dim,
            hidden_dim=dqn_config["head_hidden_dim"],
            num_actions=self.num_actions,
            num_atoms=self.num_atoms
        )

        # Z support
        self.support = torch.linspace(self.vmin, self.vmax, self.num_atoms)

        # TensorDictModule: 入力 -> logits
        self.module = TensorDictSequential(
            TensorDictModule(
                module=self._forward_logic,
                in_keys=["observation"],
                out_keys=["logits"]
            ),
            TensorDictModule(
                lambda logit: F.log_softmax(logit, dim=-2),
                in_keys=["logits"],
                out_keys=["action_value"]),
        )

        # DistributionalQValueActor: logits -> action
        self.actor = DistributionalQValueActor(
            module=self.module,
            in_keys=["observation"],
            support=self.support,
            action_space="categorical",
            action_value_key="action_value",
            make_log_softmax=False
        )

    def _forward_logic(self, obs_seq: torch.Tensor) -> torch.Tensor:
        # obs_seq # [batch_size, seq_length, observation_space]
        rnn_out, _ = self.rnn(obs_seq)
        last_feat = rnn_out[:, -1, :]
        features = self.feature_net(last_feat)
        logits = self.head(features)  # [B, A, N]
        logits = logits.permute(0, 2, 1)  # [B, N, A] に変換（DistributionalQValueActor 用）
        return logits

    def forward(self, obs_seq: torch.Tensor) -> TensorDict:
        td = TensorDict({"observation": obs_seq}, batch_size=obs_seq.shape[:1])
        td = self.actor(td)  # ここで "action" が追加される
        return td

    def get_q_values(self, obs_seq: torch.Tensor) -> torch.Tensor:
        td = TensorDict({"observation": obs_seq}, batch_size=obs_seq.shape[:1])
        logits = self.module(td)["logits"]
        probs = F.softmax(logits, dim=-1)
        q = (probs * self.support.to(logits.device)).sum(dim=-1)
        return q

    def to_input(self, x):
        for adapter in self.input_adapters:
            x = adapter(x)
        return x

    def to_action(self, probabilities):
        q_values = self.q_value_adapter(probabilities)
        actions = self.action_adapter(q_values)
        return actions
