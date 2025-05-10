#!/usr/bin/env python3
"""
Script to run the PER test and compare performance with standard DQN
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from cat_toy_env import CatToyEnv
from test_per import DQNAgent, DQNAgentPER

# Global parameters
num_iterations = 100
num_episodes_per_iteration = 1
num_steps_per_episode = 10000
update_target_steps = 10
replay_interval = 6
buffer_size = 10000
batch_size = 64

# Training function
def train_dqn(agent_dict, env, num_iterations, num_episodes_per_iteration):
    total_rewards = {agent: 0.0 for agent in env.agents}
    steps = 0
    rewards_history = {agent: [] for agent in env.agents}
    
    for iteration in range(num_iterations):
        for episode in range(num_episodes_per_iteration):
            obs = env.reset()
            prev_obs = {agent: obs for agent in env.agents}
            prev_action = {agent: None for agent in env.agents}
            prev_total_reward = {agent: 0.0 for agent in env.agents}

            for agent in env.agent_iter():
                if agent == "dummy":
                    # dummyエージェントは行動しない
                    action = None
                    env.step(action)
                    continue

                obs, total_reward, terminated, truncated, _ = env.last()
                done = terminated or truncated

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
                        for replay_agent in ["cat", "toy"]:
                            agent_dict[replay_agent].replay()

                if done:
                    action = None  # No action needed if agent is done
                    total_rewards[agent] += total_reward
                    steps += env.step_count
                else:
                    action = agent_dict[agent].act(obs)

                env.step(action)

                prev_obs[agent] = obs  # 次の状態を更新
                prev_action[agent] = action  # 次の行動を更新
                prev_total_reward[agent] = total_reward # 次の報酬を更新

        # ログ出力
        if iteration % update_target_steps == 0:
            for agent in total_rewards.keys():
                avg_reward = total_rewards[agent] / update_target_steps
                rewards_history[agent].append(avg_reward)
                print(f"Iteration {iteration}: {agent} avg reward: {avg_reward:.2f}")
            
            total_rewards = {agent: 0.0 for agent in total_rewards.keys()}
            steps = 0

        # ターゲットネットワーク更新
        if iteration % update_target_steps == 0:
            for agent in agent_dict.values():
                agent.update_target_model()
                
    return rewards_history

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

# Main function to run the experiment
def main():
    # Create environments
    env_kwargs = dict(render_mode=None, max_steps=num_steps_per_episode)
    env_learning = CatToyEnv(**env_kwargs)
    
    # Create agents with and without PER
    agent_dict_standard = {
        agent_name: DQNAgent(agent_name, env_learning)
        for agent_name in env_learning.agents
    }
    
    agent_dict_per = {
        agent_name: DQNAgentPER(agent_name, env_learning)
        for agent_name in env_learning.agents
    }
    
    # Train both agent types
    print("Training standard DQN agents...")
    rewards_standard = train_dqn(agent_dict_standard, env_learning, num_iterations, num_episodes_per_iteration)
    
    # Reset environment
    env_learning = CatToyEnv(**env_kwargs)
    
    print("Training DQN agents with PER...")
    rewards_per = train_dqn(agent_dict_per, env_learning, num_iterations, num_episodes_per_iteration)
    
    # Evaluate both agent types
    env_eval = CatToyEnv(**env_kwargs)
    
    print("Evaluating standard DQN agents...")
    mean_std_rewards_standard = evaluate_model(agent_dict_standard, env_eval, n_eval_episodes=10)
    
    print("Evaluating DQN agents with PER...")
    mean_std_rewards_per = evaluate_model(agent_dict_per, env_eval, n_eval_episodes=10)
    
    # Print results
    print("\nResults:")
    print("Standard DQN:")
    for agent, (mean, std) in mean_std_rewards_standard.items():
        print(f"  {agent}: {mean:.2f} ± {std:.2f}")
    
    print("DQN with PER:")
    for agent, (mean, std) in mean_std_rewards_per.items():
        print(f"  {agent}: {mean:.2f} ± {std:.2f}")
    
    # Plot learning curves
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("Cat Agent")
    plt.plot(rewards_standard["cat"], label="Standard DQN")
    plt.plot(rewards_per["cat"], label="DQN with PER")
    plt.xlabel("Iterations (x10)")
    plt.ylabel("Average Reward")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.title("Toy Agent")
    plt.plot(rewards_standard["toy"], label="Standard DQN")
    plt.plot(rewards_per["toy"], label="DQN with PER")
    plt.xlabel("Iterations (x10)")
    plt.ylabel("Average Reward")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("per_comparison.png")
    plt.show()

if __name__ == "__main__":
    main()