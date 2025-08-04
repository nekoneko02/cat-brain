import numpy as np
import random

class Dummy:
    def __init__(self, vel_seq_len=1):
        self.vel_seq_len = vel_seq_len
        self.vel = [0.0, 0.0]
        self.vel_seq = [[0.0, 0.0] for _ in range(vel_seq_len)]
        self.energy = -1000.0

    def get_action(self, observation):
        # ここでは単純にランダムなアクションを返す
        direction = random.choice([(0,0.5), (0.5,0), (-0.5,0), (0,-0.5)])
        self.vel = [direction[0], direction[1]]
        # 速度履歴更新
        self.vel_seq.pop(0)
        self.vel_seq.append(self.vel)
        return {"dx": direction[0], "dy": direction[1]} # 例: 上、下、左、右の4つのアクション
    def get_velocity(self):
        # 最新から過去順でflatten
        return np.array(self.vel_seq[::-1]).flatten()
    
    def get_energy(self):
        return self.energy  # 例: dummyを捕まえると負の報酬