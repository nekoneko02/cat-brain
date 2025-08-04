from .runner_base import RunnerBase
import numpy as np

class RunnerStopAndMove(RunnerBase):
    def __init__(self, move_interval, speed, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.move_interval = move_interval
        self.speed = speed
        self.counter = 0

    def get_action(self, obs):
        """
        obs: np.ndarray
            [rel_pos_norm(2), chaser_vel(2), runner_vel_seq(n), fatigue(1)]
        """
        self.counter += 1
        if (self.counter // self.move_interval) % 2 == 0:
            # 止まる
            return {'dx': 0.0, 'dy': 0.0}
        else:
            # ランダムに動く
            angle = np.random.uniform(0, 2*np.pi)
            move = self.speed * np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
            return {'dx': float(move[0]), 'dy': float(move[1])}
