import numpy as np

class ToySlow:
    def __init__(self):
        self.vel = np.array([0.0, 0.0], dtype=np.float32)
        self.energy = 100.0

    def get_action(self, observation):
        # obs: [rel_x, rel_y, chaser_vel_x, chaser_vel_y, runner_vel_x, runner_vel_y, fatigue]
        rel_pos = observation[0:2]

        # 追いかける方向（相対位置ベクトルを正規化）
        distance = np.linalg.norm(rel_pos)
        direction = 0.1 * rel_pos / (distance + 1e-8)
        self.vel = direction
        return  {"dx": direction[0], "dy": direction[1]}  # shape: [2]
    def get_velocity(self):
        return self.vel
    
    def get_energy(self):
        # トイのエネルギーを返す
        return 1000  # 例: トイのエネルギーは常に100とする