from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
from gymnasium import spaces
import numpy as np
import json
from IPython.display import clear_output
import random
import os
import time
import pygame

class CatToyEnv(AECEnv):
    metadata = {"render_modes": ["human"], "name": "cat_toy_env_v0"}

    def __init__(self, render_mode=None, max_steps=1000):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps

        with open('../cat-game/config/common.json', 'r') as f:
            config = json.load(f)

        env_config = config['environment']
        self.width = env_config['width']
        self.height = env_config['height']
        self.cat_width = env_config['cat_width']
        self.cat_height = env_config['cat_height']
        self.toy_width = env_config['toy_width']
        self.toy_height = env_config['toy_height']

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

        # ✅ PettingZoo AECEnv に必要な属性
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.step_count = 0
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Cat and Toy Game")
            self.clock = pygame.time.Clock()
    def observe(self, agent):
        return np.array([self.toy_x, self.toy_y, self.cat_x, self.cat_y], dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.dones = {a: False for a in self.agents}  # 互換性のために残してもOK
        self.infos = {a: {} for a in self.agents}
        self.step_count = 0

        self.cat_x = random.randint(0, self.width - self.cat_width)
        self.cat_y = random.randint(0, self.height - self.cat_height)
        self.toy_x = random.randint(0, self.width - self.toy_width)
        self.toy_y = random.randint(0, self.height - self.toy_height)

        return self.observe(self.agent_selection)


    def step(self, action):
        agent = self.agent_selection

        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        selected_action = self.actions[action]
        dx = selected_action["dx"]
        dy = selected_action["dy"]

        if agent == "cat":
            self.cat_x = np.clip(self.cat_x + dx, 0, self.width - self.cat_width)
            self.cat_y = np.clip(self.cat_y + dy, 0, self.height - self.cat_height)
        elif agent == "toy":
            self.toy_x = np.clip(self.toy_x + dx, 0, self.width - self.toy_width)
            self.toy_y = np.clip(self.toy_y + dy, 0, self.height - self.toy_height)

        collision = self._is_collision()
        if collision:
            self.terminations = {a: True for a in self.agents}
            self.rewards["cat"] = 100.0
            self.rewards["toy"] = -100.0
        elif self.step_count >= self.max_steps:
            self.truncations = {a: True for a in self.agents}
        else:
            distance = ((self.cat_x - self.toy_x) ** 2 + (self.cat_y - self.toy_y) ** 2) ** 0.5
            self.rewards["cat"] = -distance
            self.rewards["toy"] = distance

        # ✅ 報酬加算
        self._cumulative_rewards[agent] += self.rewards[agent]
        self.step_count += 1

        # ✅ 次のエージェントへ切り替え
        self.agent_selection = self._agent_selector.next()

        self._clear_rewards()
        
        if self.render_mode == "human":
            self.render()

    def _is_collision(self):
        return (
            self.cat_x < self.toy_x + self.toy_width and
            self.cat_x + self.cat_width > self.toy_x and
            self.cat_y < self.toy_y + self.toy_height and
            self.cat_y + self.cat_height > self.toy_y
        )

    def render(self):
        # 画面を黒で塗りつぶす
        self.screen.fill((0, 0, 0))

        # Cat と Toy の描画
        pygame.draw.rect(self.screen, (255, 0, 0), (self.cat_x, self.cat_y, self.cat_width, self.cat_height))  # 赤色で猫
        pygame.draw.rect(self.screen, (0, 255, 0), (self.toy_x, self.toy_y, self.toy_width, self.toy_height))  # 緑色でおもちゃ

        # 画面更新
        pygame.display.flip()

        # 1フレームの間隔を設定
        self.clock.tick(30)  # 30 FPS

    def close(self):
        pygame.quit()  # pygameを終了
