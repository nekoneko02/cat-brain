import numpy as np

class Toy:
    def __init__(self, vel_seq_len=1):
        self.vel_seq_len = vel_seq_len
        self.vel = np.array([0.0, 0.0], dtype=np.float32)
        self.vel_seq = [np.array([0.0, 0.0], dtype=np.float32) for _ in range(vel_seq_len)]
        self.energy = 1000.0

    def get_action(self, observation):
        # obs: [rel_x, rel_y, chaser_vel_x, chaser_vel_y, runner_vel_x, runner_vel_y, fatigue]
        rel_pos = observation[0:2]

        # 追いかける方向（相対位置ベクトルを正規化）
        distance = np.linalg.norm(rel_pos)
        direction = 0.5 * rel_pos / (distance + 1e-8)
        self.vel = direction
        # 速度履歴更新
        self.vel_seq.pop(0)
        self.vel_seq.append(direction)
        return  {"dx": direction[0], "dy": direction[1]}  # shape: [2]
    def get_velocity(self):
        # 最新から過去順でflatten
        return np.array(self.vel_seq[::-1]).flatten()
    
    def get_energy(self):
        # トイのエネルギーを返す
        return self.energy  # 例: トイのエネルギーは常に100とする


class ToySlow:
    def __init__(self, vel_seq_len=1):
        self.vel_seq_len = vel_seq_len
        self.vel = np.array([0.0, 0.0], dtype=np.float32)
        self.vel_seq = [np.array([0.0, 0.0], dtype=np.float32) for _ in range(vel_seq_len)]
        self.energy = 1000.0

    def get_action(self, observation):
        # obs: [rel_x, rel_y, chaser_vel_x, chaser_vel_y, runner_vel_x, runner_vel_y, fatigue]
        rel_pos = observation[0:2]

        # 追いかける方向（相対位置ベクトルを正規化）
        distance = np.linalg.norm(rel_pos)
        direction = 0.1 * rel_pos / (distance + 1e-8)
        self.vel = direction
        # 速度履歴更新
        self.vel_seq.pop(0)
        self.vel_seq.append(direction)
        return  {"dx": direction[0], "dy": direction[1]}  # shape: [2]
    def get_velocity(self):
        # 最新から過去順でflatten
        return np.array(self.vel_seq[::-1]).flatten()
    
    def get_energy(self):
        # トイのエネルギーを返す
        return self.energy  # 例: トイのエネルギーは常に100とする