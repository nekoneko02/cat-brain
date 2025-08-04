from .runner_base import RunnerBase
import numpy as np

class RunnerStop(RunnerBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_action(self, obs):
        # 常に止まる
        return {'dx': 0.0, 'dy': 0.0}
