import random

class Dummy:
    def __init__(self):
        pass

    def get_action(self, observation):
        # ここでは単純にランダムなアクションを返す
        direction = random.choice([(0,1), (1,0), (-1,0), (0,-1)])
        return {"dx": direction[0], "dy": direction[1]} # 例: 上、下、左、右の4つのアクション
    
    def get_energy(self):
        return -100  # 例: dummyを捕まえると負の報酬