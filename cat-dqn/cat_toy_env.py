import gymnasium as gym 
from gymnasium import spaces
import numpy as np
import random
import json

class CatToyEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, max_steps=1000):
        super().__init__()

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

        # 観測空間を設定ファイルから動的に構築
        obs_config = config['observation_space']
        self.observation_space = spaces.Box(
            low=obs_config['low'],
            high=obs_config['high'],
            shape=tuple(obs_config['shape']),
            dtype=getattr(np, obs_config['dtype'])
        )
        self.action_space = spaces.Discrete(len(self.actions))  # アクション数を取得

        # ✅ 観測空間を Dict から Box(連結済み) に変更：toy_x, toy_y, cat_x, cat_y → shape=(4,)
        self.observation_space = spaces.Box(low=0, high=max(self.width, self.height), shape=(4,), dtype=np.float32)

        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.window_size = (self.width, self.height)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cat_x = random.randint(0, self.width - self.cat_width)
        self.cat_y = random.randint(0, self.height - self.cat_height)
        self.toy_x = random.randint(0, self.width - self.toy_width)
        self.toy_y = random.randint(0, self.height - self.toy_height)
        self.step_count = 0

        observation = self._get_obs()
        info = {}
        if self.render_mode == "human":
            self._render_frame()
        return observation, info

    def step(self, action):
        self._cat_move(action)
        reward = self._calculate_reward()
        terminated = self._is_done()
        truncated = False
        observation = self._get_obs()
        info = {}
        self.step_count += 1

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        # ✅ Dict ではなく、単一の float32 配列を返す
        return np.array(
            [self.toy_x, self.toy_y, self.cat_x, self.cat_y],
            dtype=np.float32
        )

    def _cat_move(self, action):
        # ✅ 外部JSONファイルから読み込んだアクションを使用
        selected_action = next((a for a in self.actions if a['id'] == action), None)
        if selected_action:
            self.cat_x += selected_action['dx']
            self.cat_y += selected_action['dy']

        # 境界チェック
        self.cat_x = max(0, min(self.cat_x, self.width - self.cat_width))
        self.cat_y = max(0, min(self.cat_y, self.height - self.cat_height))

        self.cat_x = max(0, min(self.cat_x, self.width - self.cat_width))
        self.cat_y = max(0, min(self.cat_y, self.height - self.cat_height))

    def _calculate_reward(self):
        distance = ((self.cat_x - self.toy_x)**2 + (self.cat_y - self.toy_y)**2)**0.5
        reward = 100 if distance < 10 else -distance
        if (self.cat_x <= 0 or self.cat_x >= self.width or self.cat_y <= 0 or self.cat_y >= self.height):
            reward -= 10
        reward -= 0.1
        return reward

    def _is_done(self):
        distance = ((self.cat_x - self.toy_x)**2 + (self.cat_y - self.toy_y)**2)**0.5
        return self.step_count >= self.max_steps or distance < 10

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        import pygame
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(self.window_size)
        canvas.fill((255, 255, 255))
        pygame.draw.rect(canvas, (255, 0, 0), pygame.Rect(self.toy_x, self.toy_y, self.toy_width, self.toy_height))
        pygame.draw.rect(canvas, (0, 0, 255), pygame.Rect(self.cat_x, self.cat_y, self.cat_width, self.cat_height))

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
