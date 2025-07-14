import numpy as np


def train_dqn(agent, env, config):
    total_step = 0
    num_episodes = 0
    training_steps = config["training_steps"]
    replay_interval = config["replay_interval"]
    update_target_steps = config["update_target_steps"]
    batch_size = config["batch_size"]

    total_reward = 0.0
    steps = 0
    while total_step < training_steps:
        obs, _ = env.reset()
        done = False

        while not done:
            steps += 1

            option, action = agent.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            
            # 前回行動の結果が今回のループで得られたので、ここで保存できる
            agent.store_experience(
                obs,         # s
                option,      # a
                reward,      # r (現在のループで得られた報酬)
                next_obs,    # s' (次状態)
                float(done)  # done
            )
            # ここでreplayを行う
            if env.step_count % replay_interval == 0:
                agent.replay(batch_size)

            if done or env.step_count % 1000 == 0:
                formated_obs = ", ".join([f"{x:.2f}" for x in obs])
                formated_reward = f"{(reward):+7.2f}"
                print(f"steps {env.step_count:>5}, reward {formated_reward}, state is {formated_obs}")
            
            obs = next_obs

            # ターゲットネットワーク更新
            if env.step_count % (update_target_steps * 4000) == 0:
                agent.update_target_model()
        num_episodes += 1
        # ログ出力
        print(f"+++++++ Episode {num_episodes}: " + ", ".join([f"{total_reward / update_target_steps:.2f}"]), steps / update_target_steps)
        total_reward = 0.0
        steps = 0

def evaluate_model(agent, eval_env, n_eval_episodes=10):
    reward_sum = []

    for _ in range(n_eval_episodes):
        env = eval_env  # 環境がreset可能で、内部状態が共有でないと仮定
        obs, _ = env.reset()
        episode_reward = 0.0
        done = False
        while not done:
            option, action = agent.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            obs = next_obs
            episode_reward += reward  # 各agentごとに報酬を記録

        reward_sum.append(episode_reward)

    # 統計量（平均・標準偏差）を返す
    mean_std_reward = (np.mean(reward), np.std(reward))

    return mean_std_reward
