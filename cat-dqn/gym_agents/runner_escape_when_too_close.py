from .runner_base import RunnerBase
import numpy as np

class RunnerEscapeWhenTooClose(RunnerBase):
    def __init__(self, threshold, speed, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold
        self.speed = speed

    def get_action(self, obs):
        # ねこから遠ざかる
        dx, dy = obs[0], obs[1]
        dist = np.sqrt(dx**2 + dy**2)
        if dist < self.threshold:
            # ねこから遠ざかる方向に移動
            norm = np.sqrt(dx**2 + dy**2)
            if norm == 0:
                return {'dx': 0.0, 'dy': 0.0}
            move = self.speed * np.array([dx, dy]) / norm
            return {'dx': float(move[0]), 'dy': float(move[1])}
        else:
            # ランダム移動
            angle = np.random.uniform(0, 2*np.pi)
            move = self.speed * np.array([np.cos(angle), np.sin(angle)])
            return {'dx': float(move[0]), 'dy': float(move[1])}
