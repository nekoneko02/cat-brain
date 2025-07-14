import numpy as np

class Toy:
    def __init__(self):
        pass

    def get_action(self, observation):
        # obs: [cat_x, cat_y, toy_x, toy_y, energy]
        cat_pos = np.array(observation[0:2])
        toy_pos = np.array(observation[2:4])

        # catから遠ざかる方向
        distance_vec = -(cat_pos - toy_pos)
        distance = np.linalg.norm(distance_vec)
        direction = (distance_vec / distance) * 0.5  # 0.7 is a scaling factor to control the speed

        return {"dx": direction[0], "dy": direction[1]}
    
    def get_energy(self):
        # トイのエネルギーを返す
        return 100  # 例: トイのエネルギーは常に100とする