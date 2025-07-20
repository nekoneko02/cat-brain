import importlib
import random
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchrl.objectives import DistributionalDQNLoss
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, TensorDictPrioritizedReplayBuffer

import replay_buffer
importlib.reload(replay_buffer)

from replay_buffer import SequenceTensorDictPrioritizedReplayBuffer

import cat_actions
importlib.reload(cat_actions)


class CatAgent:
    def __init__(self, dqn, dqn_config, agent_config, device="cpu", epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.gamma = agent_config["discount_rate"]
        self.device = device
        self.action_space = agent_config["action_space"]
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.model = dqn
        self.cat_actions = [
            cat_actions.chase.Chase(),
            cat_actions.stop.Stop()
        ]

        self.optimizer = optim.Adam(self.model.parameters(), lr=agent_config["learning_rate"])

        self.loss_module = DistributionalDQNLoss(
            value_network=self.model.actor,
            delay_value=True,
            gamma=self.gamma
        )

        buffer_config = agent_config["buffer"]
        self.seq_obs = deque(maxlen=dqn_config["rnn"]["sequence_length"])

        self.memory = SequenceTensorDictPrioritizedReplayBuffer(
            storage=LazyTensorStorage(buffer_config["size"], device=device),
            alpha=buffer_config["alpha"],
            beta=buffer_config["beta"],
            sequence_length=dqn_config["rnn"]["sequence_length"],
        )

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.add(TensorDict({
            'state': torch.tensor(state, dtype=torch.float32, device=self.device),
            'action': torch.tensor([action], dtype=torch.long, device=self.device),
            'reward': torch.tensor([reward], dtype=torch.float32, device=self.device),
            'next_state': torch.tensor(next_state, dtype=torch.float32, device=self.device),
            'done': torch.tensor([done], dtype=torch.float32, device=self.device),
            'td_error': torch.tensor(1.0, dtype=torch.float32, device=self.device)
        }))

    def act(self, state):
        if random.random() <= self.epsilon:
            option = self.action_space.sample()
        else:
            x = self.model.to_input(state)
            with torch.no_grad():
                td = self.model.actor(TensorDict({"observation": x}, batch_size=x.shape[:1]))
                option = td["action"]
        action = self.cat_actions[option.item()](torch.Tensor(state))
        return option, {
            "dx": action[0].item(),
            "dy": action[1].item()
        }

    def _get_sarsa(self, batch_size, return_info=True):
        batch, info = self.memory.sample(batch_size, return_info=return_info)
        indices, weights = info['index'], info['_weight']
        weights = torch.FloatTensor(weights).to(self.device)

        data = TensorDict({
            "observation": batch['state'],
            "action": batch['action'][:, -1, :],
            "next": TensorDict({
                "observation": batch['next_state'],
                "reward": batch['reward'][:, -1, :],
                "done": batch['done'][:, -1, :],
                "terminated": batch['done'][:, -1, :]
            }),
            "steps_to_next_obs": torch.ones(64, 1),
            "td_error": weights.unsqueeze(-1),
        }, batch_size=[batch_size]).to(self.device)
        return indices, data

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        indices, data = self._get_sarsa(batch_size)

        loss_td = self.loss_module(data)
        loss = loss_td["loss"]
        td_error = torch.ones(64, 1) # Todo: loss_tdにtd_errorが含まれていないため、PERを停止中。
        priority = torch.clamp(td_error, min=1.0, max=1e3)
        self.memory.update_priority(indices, priority)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, filepath):
        checkpoint = {"model_state_dict": self.model.state_dict()}
        torch.save(checkpoint, filepath)

    def load_model(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
