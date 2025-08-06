from .runner_base import RunnerBase
import numpy as np

class Enemy(RunnerBase):
    def __init__(self, speed, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.speed = speed

    def _get_action(self, obs):
        # Catに近づく
        dx, dy = -obs[0], -obs[1]
        norm = np.sqrt(dx**2 + dy**2)
        if norm == 0:
            return {'dx': 0.0, 'dy': 0.0}
        move = self.speed * np.array([dx, dy]) / norm
        return {'dx': float(move[0]), 'dy': float(move[1])}
