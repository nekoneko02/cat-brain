import gymnasium as gym 
from gymnasium import spaces
import numpy as np
import random

class CatToyEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, width=800, height=600, max_steps=100, cat_speed=2, cat_width=30, cat_height=30, toy_width=20, toy_height=20):
        super().__init__()
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.cat_speed = cat_speed
        self.cat_width = cat_width
        self.cat_height = cat_height
        self.toy_width = toy_width
        self.toy_height = toy_height

        self.action_space = spaces.Discrete(4)

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
        if action == 0:
            self.cat_y -= self.cat_speed
        elif action == 1:
            self.cat_y += self.cat_speed
        elif action == 2:
            self.cat_x -= self.cat_speed
        elif action == 3:
            self.cat_x += self.cat_speed

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
