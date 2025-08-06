import random
import time

import numpy as np
from gymnasium import spaces, Env
from IPython.display import clear_output

RUNNNER_DISTANCE_THRESHOLD = 1000

class CatToyEnv(Env):
    metadata = {"render_modes": ["human"], "name": "cat_toy_env_v0"}

    def __init__(
        self,
        render_mode=None,
        chaser=None,
        runners=None,
        reset_interval=2000
    ):
        super().__init__()
        self.render_mode = render_mode
        if self.render_mode == "human":
            self.render_width = 800
            self.render_height = 800

        self.reset_interval = reset_interval       

        # agent設定
        self.Chaser = chaser
        self.Runners = runners

        # 環境サイズ等はrunner/chaserから取得する前提
        tmp_chaser = self.Chaser()
        tmp_runner = self.Runners[0][0]()
        # 位置情報は仮で0,0
        rel_pos_shape = (2,)
        chaser_vel_shape = np.array(tmp_chaser.get_velocity()).shape
        runner_vel_shape = np.array(tmp_runner.get_velocity()).shape
        fatigue_shape = (1,)
        obs_shape = (rel_pos_shape[0] + chaser_vel_shape[0] + runner_vel_shape[0] + fatigue_shape[0],)
        self.observation_space = spaces.Box(
            low=-1000,
            high=1000,
            shape=obs_shape,
            dtype=np.float32,
        )
        self.action_space = tmp_chaser.get_action_space()

    def _get_obs(self):
        # 相対位置（Runner - Chaser, tanh正規化）
        chaser_pos = np.array(self.positions[self.chaser], dtype=np.float32)
        runner_pos = np.array(self.positions[self.current_runner], dtype=np.float32)
        rel_pos = runner_pos - chaser_pos
        rel_pos_norm = np.tanh(rel_pos / 2000.0)

        # Chaser速度ベクトル
        chaser_vel = np.array(self.chaser.get_velocity(), dtype=np.float32)
        # Runner速度履歴（flatten済み）
        runner_vel_seq = np.array(self.current_runner.get_velocity(), dtype=np.float32)

        # Chaser疲労度（0.0～1.0）
        fatigue = self.chaser.get_fatigue()

        obs = np.concatenate([rel_pos_norm, chaser_vel, runner_vel_seq, [fatigue]]).astype(np.float32)
        return obs

    def _init_runner(self):
        # (Factory, rate)リストからrateに従いサンプリング
        factories, rates = zip(*self.Runners)
        idx = np.random.choice(len(factories), p=rates)
        self.current_runner = factories[idx]()
        self.info["current_runner"] = str(self.current_runner)
        
        # chaserの位置取得
        chaser_pos = np.array(self.positions[self.chaser], dtype=np.float32)

        # 距離100~300の範囲でランダムに決定
        distance = np.random.uniform(100, 300)
        # ランダムな方向ベクトル（大きさ1）生成
        angle = np.random.uniform(0, 2 * np.pi)
        direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        offset = direction * distance
        runner_pos = chaser_pos + offset
        self.positions[self.current_runner] = [float(runner_pos[0]), float(runner_pos[1])]

        self.prev_distance = self.squared_distance(self.chaser, self.current_runner)

    def _init_chaser(self):
        self.chaser = self.Chaser()
        self.positions[self.chaser] = [0,0]

    def reset(self, seed=None, options=None):
        self.chaser = None
        self.current_runner = None
        self.positions = {}
        self.reward = 0
        self.terminated = False
        self.truncated = False
        self.info = {}

        self._init_chaser()
        self._init_runner()

        self.step_count = 0

        obs = self._get_obs()
        return obs, self.info

    def step(self, action):
        # action: int (cat_actionsのindex)
        self.step_count += 1

        # cat_actionsのindexからaction dictを取得
        self._step_runners()
        reward, terminated, truncated, is_collision = self._step_cat(action)

        obs = self._get_obs()

        # runnerのリセット判定をRunnerBaseのis_resetで一元化（cat_pos, toy_pos, step_countを渡す）
        cat_pos = self.positions[self.chaser]
        toy_pos = self.positions[self.current_runner]
        if self.current_runner.is_reset(cat_pos, toy_pos):
            self._init_runner()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, self.info

    def _step_cat(self, action):
        reward = 1.0
        terminated = False
        truncated = False

        # catの行動
        action = self.chaser.get_action(action, self._get_obs())
        self._move_agent(self.chaser, action)

        distance = self.squared_distance(self.chaser, self.current_runner)
        
        cat_pos = self.positions[self.chaser]
        toy_pos = self.positions[self.current_runner]
        is_collision = self.current_runner.is_caught(cat_pos, toy_pos)

        # 最適行動によるボーナス
        if self.current_runner.get_energy() > 0 and distance < self.prev_distance:
            reward += 1
        if self.current_runner.get_energy() <= 0 and distance >= self.prev_distance:
            reward += 1
        
        self.prev_distance = distance
        
        # 衝突判定
        if is_collision:
            print(f"cat catch toy! {self.info}")
            toy_energy = self.current_runner.get_energy()
            self.chaser.eat(toy_energy)
            reward += 10 * toy_energy / abs(toy_energy)  # トイのエネルギーを報酬に変換
            
        
        # エネルギー切れ
        if self.chaser.energy <= 0:
            print("cat is tired")
            truncated = True
            reward = -10
        return reward, terminated, truncated, is_collision
    def _step_runners(self):
        action = self.current_runner.get_action(self._get_obs())
        self._move_agent(self.current_runner, action)
        
    def _move_agent(self, agent, action):
        dx, dy = action["dx"], action["dy"]
        x, y = self.positions[agent]
        new_x = x + dx
        new_y = y + dy
        self.positions[agent][0] = new_x
        self.positions[agent][1] = new_y

    def squared_distance(self, agent1, agent2):
        agent1_x, agent1_y = self.positions[agent1]
        agent2_x, agent2_y = self.positions[agent2]
        distance = (agent1_x - agent2_x) ** 2 + (agent1_y - agent2_y) ** 2
        return distance

    def render(self):
        if self.step_count % 30 != 0:
            return
        grid_size = 30
        scale = grid_size / max(self.render_width, self.render_height)
        grid = [["." for _ in range(grid_size)] for _ in range(grid_size)]

        original_cat_x, original_cat_y = self.positions[self.chaser]
        cat_x = int(original_cat_x * scale)
        cat_y = int(original_cat_y * scale)
        original_toy_x, original_toy_y = self.positions[self.current_runner]
        toy_x = int(original_toy_x * scale)
        toy_y = int(original_toy_y * scale)

        # render専用の中心座標を管理
        if not hasattr(self, "_render_center"):
            self._render_center = [cat_x, cat_y]

        center_x, center_y = self._render_center
        min_x = int(center_x - grid_size // 2)
        min_y = int(center_y - grid_size // 2)

        # Catが枠外に出たら中心座標を更新
        cat_gx = int(cat_x - min_x)
        cat_gy = int(cat_y - min_y)
        if not (0 <= cat_gx < grid_size and 0 <= cat_gy < grid_size):
            self._render_center = [cat_x, cat_y]
            center_x, center_y = self._render_center
            min_x = int(center_x - grid_size // 2)
            min_y = int(center_y - grid_size // 2)
            cat_gx = int(cat_x - min_x)
            cat_gy = int(cat_y - min_y)

        def to_grid_coords(x, y):
            gx = int(x - min_x)
            gy = int(y - min_y)
            if 0 <= gx < grid_size and 0 <= gy < grid_size:
                return gx, gy
            return None, None

        toy_gx, toy_gy = to_grid_coords(toy_x, toy_y)
        if 0 <= cat_gx < grid_size and 0 <= cat_gy < grid_size:
            if cat_gx == toy_gx and cat_gy == toy_gy:
                grid[cat_gy][cat_gx] = "C&T"
            else:
                grid[cat_gy][cat_gx] = "C"
        if toy_gx is not None and toy_gy is not None and (cat_gx != toy_gx or cat_gy != toy_gy):
            grid[toy_gy][toy_gx] = "T"
        clear_output(wait=True)
        for row in reversed(grid):
            print(" ".join(row))
        print("-" * (2 * grid_size))
        print(
            f"count: {self.step_count}, positions: cat: ({original_cat_x}, {original_cat_y}), toy: ({original_toy_x}, {original_toy_y}))"
        )
        time.sleep(0.01)

    def close(self):
        pass
