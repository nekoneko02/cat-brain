import numpy as np

def train_dqn(agent_dict, env, config):
    num_iterations = config["num_iterations"]
    num_episodes_per_iteration = config["num_episodes_per_iteration"]
    replay_interval = config["replay_interval"]
    update_target_steps = config["update_target_steps"]
    batch_size = config["batch_size"]

    total_rewards = {agent: 0.0 for agent in agent_dict.keys()}
    steps = 0
    for iteration in range(num_iterations):
        for episode in range(num_episodes_per_iteration):
            obs = env.reset()
            prev_obs = {agent: obs for agent in agent_dict.keys()} # 前回の観測を保存
            prev_action = {agent: None for agent in agent_dict.keys()}
            prev_total_reward = {agent: 0.0 for agent in agent_dict.keys()}

            for agent in env.agent_iter():
                if agent == "dummy":
                    # dummyエージェントは行動しない
                    action = None
                    env.step(action)
                    continue

                obs, total_reward, terminated, truncated, _ = env.last()
                done = terminated or truncated

                if done:
                    action = None  # No action needed if agent is done
                    total_rewards[agent] += total_reward
                    steps += env.step_count
                else:
                    action = agent_dict[agent].act(obs)
                    agent_dict[agent].reset_hidden_state() # 行動を選択するたびにノイズをリセット

                env.step(action)

                if prev_action[agent] is not None:
                    # 前回行動の結果が今回のループで得られたので、ここで保存できる
                    agent_dict[agent].store_experience(
                        prev_obs[agent],         # s
                        prev_action[agent],      # a
                        total_reward - prev_total_reward[agent],      # r (現在のループで得られた報酬)
                        obs,                     # s' (次状態)
                        float(terminated)              # done
                    )
                    # ここでreplayを行う
                    if env.step_count % replay_interval == 0:
                        for replay_agent in agent_dict.keys():
                            agent_dict[replay_agent].replay(batch_size)

                if done or env.step_count % 1000 == 0:
                    print(f"{agent} with steps {env.step_count}, reward {total_reward - prev_total_reward[agent]: 2f}, action: {prev_action}, state is {obs}")

                prev_action[agent] = action  # 次の行動を更新
                prev_total_reward[agent] = total_reward # 次の報酬を更新
                prev_obs[agent] = obs

        # ログ出力
        if iteration % update_target_steps == 0:
            print(f"+++++++ Iteration {iteration}: " + ", ".join([f"{a}: {r / update_target_steps:.2f}" for a, r in total_rewards.items()]), steps / update_target_steps)
            total_rewards = {agent: 0.0 for agent in total_rewards.keys()}
            steps = 0

        # ターゲットネットワーク更新
        if iteration % update_target_steps == 0:
            for agent in agent_dict.values():
                agent.update_target_model()

def evaluate_model(agent_dict, eval_env, n_eval_episodes=10):
    reward_sums = {agent_name: [] for agent_name in agent_dict.keys()}

    for _ in range(n_eval_episodes):
        env = eval_env  # 環境がreset可能で、内部状態が共有でないと仮定
        env.reset()
        episode_rewards = {agent_name: 0.0 for agent_name in agent_dict.keys()}

        for agent in env.agent_iter():
            if agent == "dummy":
                # dummyエージェントは行動しない
                action = None
                env.step(action)
                continue
            obs, reward, termination, truncation, info = env.last()
            done = termination or truncation

            if done:
                action = None  # 終了したら行動不要
            else:
                action = agent_dict[agent].act(obs)  # 各エージェントに行動させる
                agent_dict[agent].reset_hidden_state()  # 行動を選択するたびにノイズをリセット

            env.step(action)
            episode_rewards[agent] += reward  # 各agentごとに報酬を記録

        for agent_name in reward_sums:
            reward_sums[agent_name].append(episode_rewards[agent_name])

    # 統計量（平均・標準偏差）を返す
    mean_std_rewards = {
        agent: (np.mean(rewards), np.std(rewards))
        for agent, rewards in reward_sums.items()
    }

    return mean_std_rewards
