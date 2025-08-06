from .runner_base import RunnerBase
import numpy as np

class RunnerSnake(RunnerBase):
    def __init__(self, speed, amplitude, freq, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.speed = speed
        self.amplitude = amplitude
        self.freq = freq
        self.t = 0

    def _get_action(self, obs):
        """
        RunnerSnakeは、猫から遠ざかる方向に進みつつ、蛇行（ジグザグ）する動きを加えます。
        - 基本方向はRunnerからCatへのベクトルの逆方向（遠ざかる）
        - その方向ベクトルに対して、sin波で回転を加え、蛇行成分を作る
        - amplitude: 蛇行の強さ（回転角の最大値）
        - freq: 蛇行の周期（sin波の周波数）
        - speed: 移動速度
        obs: [rel_pos_norm(2), chaser_vel(2), runner_vel_seq(n), fatigue(1)]
        """
        self.t += 1
        # Catの位置から遠ざかる方向ベクトル
        dx = obs[0]
        dy = obs[1]
        norm = np.sqrt(dx**2 + dy**2)
        if norm == 0:
            base_dir = np.array([0.0, 0.0])
        else:
            base_dir = np.array([dx, dy]) / norm
        # 蛇行成分（sin波で回転角を生成）
        angle = np.sin(self.t * self.freq) * self.amplitude
        rot_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        move_dir = rot_matrix @ base_dir
        move = self.speed * move_dir
        return {'dx': float(move[0]), 'dy': float(move[1])}
