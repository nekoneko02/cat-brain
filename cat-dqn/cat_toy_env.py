from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from gymnasium import spaces
import numpy as np
import json
import random

class CatToyEnv(AECEnv):
    metadata = {"render_modes": ["human"], "name": "cat_toy_env_v0"}

    def __init__(self, render_mode=None, max_steps=1000):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps

        # ✅ 設定ファイルを読み込む
        with open('../cat-game/config/common.json', 'r') as f:
            config = json.load(f)

        # 環境パラメータを設定ファイルから取得
        env_config = config['environment']
        self.width = env_config['width']
        self.height = env_config['height']
        self.cat_width = env_config['cat_width']
        self.cat_height = env_config['cat_height']
        self.toy_width = env_config['toy_width']
        self.toy_height = env_config['toy_height']

        # アクション定義を読み込む
        self.actions = config['actions']

        self.possible_agents = ['cat', 'toy']
        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.observation_spaces = {
            "cat": spaces.Box(low=0, high=max(self.width, self.height), shape=(4,), dtype=np.float32),
            "toy": spaces.Box(low=0, high=max(self.width, self.height), shape=(4,), dtype=np.float32),
        }
        self.action_spaces = {
            "cat": spaces.Discrete(len(self.actions)),
            "toy": spaces.Discrete(len(self.actions)),
        }

        self.step_count = 0

    def observe(self, agent):
        return np.array([self.toy_x, self.toy_y, self.cat_x, self.cat_y], dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.rewards = {a: 0.0 for a in self.agents}
        self.dones = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self.step_count = 0

        self.cat_x = random.randint(0, self.width - self.cat_width)
        self.cat_y = random.randint(0, self.height - self.cat_height)
        self.toy_x = random.randint(0, self.width - self.toy_width)
        self.toy_y = random.randint(0, self.height - self.toy_height)

    def step(self, action):
        if self.dones[self.agent_selection]:
            self._was_done_step(action)
            return

        agent = self.agent_selection
        selected_action = self.actions[action]
        dx = selected_action["dx"]
        dy = selected_action["dy"]

        if agent == "cat":
            self.cat_x = np.clip(self.cat_x + dx, 0, self.width - self.cat_width)
            self.cat_y = np.clip(self.cat_y + dy, 0, self.height - self.cat_height)
        elif agent == "toy":
            self.toy_x = np.clip(self.toy_x + dx, 0, self.width - self.toy_width)
            self.toy_y = np.clip(self.toy_y + dy, 0, self.height - self.toy_height)

        # 次のエージェントに切り替え
        self.agent_selection = self._agent_selector.next()
        self.step_count += 1

        # 追いついたら終了
        collision = self._is_collision()
        if collision:
            self.dones = {a: True for a in self.agents}
            self.rewards["cat"] = 100.0
            self.rewards["toy"] = -100.0
        elif self.step_count >= self.max_steps:
            self.dones = {a: True for a in self.agents}
        else:
            distance = ((self.cat_x - self.toy_x) ** 2 + (self.cat_y - self.toy_y) ** 2) ** 0.5
            self.rewards["cat"] = -distance
            self.rewards["toy"] = distance

    def _is_collision(self):
        return (
            self.cat_x < self.toy_x + self.toy_width and
            self.cat_x + self.cat_width > self.toy_x and
            self.cat_y < self.toy_y + self.toy_height and
            self.cat_y + self.cat_height > self.toy_y
        )

    def render(self):
        print(f"Cat: ({self.cat_x}, {self.cat_y}), Toy: ({self.toy_x}, {self.toy_y})")
