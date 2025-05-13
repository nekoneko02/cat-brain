import torch
import numpy as np
import os
from agent import DQNAgent

def train_dqn(agent_dict, env, num_iterations, num_episodes_per_iteration, 
              update_target_steps=10, replay_interval=6):
    """
    Train DQN agents in the environment.
    
    Args:
        agent_dict (dict): Dictionary of agents
        env: Environment to train in
        num_iterations (int): Number of iterations to train
        num_episodes_per_iteration (int): Number of episodes per iteration
        update_target_steps (int): Steps between target network updates
        replay_interval (int): Steps between replay updates
        
    Returns:
        dict: Dictionary of total rewards per agent
    """
    total_rewards = {agent: 0.0 for agent in agent_dict.keys()}
    steps = 0
    
    for iteration in range(num_iterations):
        for episode in range(num_episodes_per_iteration):
            obs = env.reset()
            prev_obs = {agent: obs for agent in agent_dict.keys()}  # Store previous observations
            prev_action = {agent: None for agent in agent_dict.keys()}
            prev_total_reward = {agent: 0.0 for agent in agent_dict.keys()}

            for agent in env.agent_iter():
                if agent == "dummy":
                    # Dummy agent doesn't take actions
                    action = None
                    env.step(action)
                    continue

                obs, total_reward, terminated, truncated, _ = env.last()
                done = terminated or truncated

                if prev_action[agent] is not None:
                    # Store experience from previous action
                    agent_dict[agent].store_experience(
                        prev_obs[agent],         # s
                        prev_action[agent],      # a
                        total_reward - prev_total_reward[agent],  # r (reward from this step)
                        obs,                     # s' (next state)
                        float(terminated)        # done
                    )
                    # Perform replay at intervals
                    if env.step_count % replay_interval == 0:
                        for replay_agent in agent_dict.keys():
                            agent_dict[replay_agent].replay()

                if done or env.step_count % 1000 == 0:
import logging  # Import the logging module

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

                obs, total_reward, terminated, truncated, _ = env.last()
                done = terminated or truncated

                if prev_action[agent] is not None:
                    # Store experience from previous action
                    agent_dict[agent].store_experience(
                        prev_obs[agent],         # s
                        prev_action[agent],      # a
                        total_reward - prev_total_reward[agent],  # r (reward from this step)
                        obs,                     # s' (next state)
                        float(terminated)        # done
                    )
                    # Perform replay at intervals
                    if env.step_count % replay_interval == 0:
                        for replay_agent in agent_dict.keys():
                            agent_dict[replay_agent].replay()

                if done or env.step_count % 1000 == 0:
                    logging.info(f"{agent} with steps {env.step_count}, reward {total_reward - prev_total_reward[agent]:.2f}, action: {prev_action}, state is {obs}")

                if done:
                    action = None  # No action needed if agent is done
                    total_rewards[agent] += total_reward
                    steps += env.step_count
                else:
                    action = agent_dict[agent].act(obs)
                    agent_dict[agent].reset_hidden_state()  # Reset noise after selecting action

                env.step(action)

                prev_action[agent] = action  # Update for next iteration
                prev_total_reward[agent] = total_reward
                prev_obs[agent] = obs

        # Log output
        if iteration % update_target_steps == 0:
            logging.info(f"+++++++ Iteration {iteration}: " + ", ".join([f"{a}: {r / update_target_steps:.2f}" for a, r in total_rewards.items()]) + f", Steps: {steps / update_target_steps}")
            total_rewards = {agent: 0.0 for agent in total_rewards.keys()}
            steps = 0

                if done:
                    action = None  # No action needed if agent is done
                    total_rewards[agent] += total_reward
                    steps += env.step_count
                else:
                    action = agent_dict[agent].act(obs)
                    agent_dict[agent].reset_hidden_state()  # Reset noise after selecting action

                env.step(action)

                prev_action[agent] = action  # Update for next iteration
                prev_total_reward[agent] = total_reward
                prev_obs[agent] = obs

        # Log output
        if iteration % update_target_steps == 0:
            print(f"+++++++ Iteration {iteration}: " + ", ".join([f"{a}: {r / update_target_steps:.2f}" for a, r in total_rewards.items()]), steps / update_target_steps)
            total_rewards = {agent: 0.0 for agent in total_rewards.keys()}
            steps = 0

        # Update target networks
        if iteration % update_target_steps == 0:
            for agent in agent_dict.values():
                agent.update_target_model()
                
    return total_rewards

def evaluate_model(agent_dict, eval_env, n_eval_episodes=10):
    """
    Evaluate trained agents in the environment.
    
    Args:
        agent_dict (dict): Dictionary of agents
        eval_env: Environment for evaluation
        n_eval_episodes (int): Number of episodes to evaluate
        
    Returns:
        dict: Dictionary of mean and std rewards per agent
    """
    reward_sums = {agent_name: [] for agent_name in agent_dict.keys()}

    for _ in range(n_eval_episodes):
        eval_env.reset()
        episode_rewards = {agent_name: 0.0 for agent_name in agent_dict.keys()}

        for agent in eval_env.agent_iter():
            if agent == "dummy":
                # Dummy agent doesn't take actions
                action = None
                eval_env.step(action)
                continue
                
            obs, reward, termination, truncation, info = eval_env.last()
            done = termination or truncation

            if done:
                action = None  # No action needed if agent is done
            else:
                action = agent_dict[agent].act(obs)
                agent_dict[agent].reset_hidden_state()  # Reset noise after selecting action

            eval_env.step(action)
            episode_rewards[agent] += reward  # Record reward for each agent

        for agent_name in reward_sums:
            reward_sums[agent_name].append(episode_rewards[agent_name])

    # Calculate statistics (mean and std)
    mean_std_rewards = {
        agent: (np.mean(rewards), np.std(rewards))
        for agent, rewards in reward_sums.items()
    }

    return mean_std_rewards

def save_dqn(agent_dict, base_path="models"):
    """
    Save agent models to files.
    
    Args:
        agent_dict (dict): Dictionary of agents
        base_path (str): Directory to save models in
    """
    os.makedirs(base_path, exist_ok=True)
    for agent_name, agent in agent_dict.items():
        filepath = os.path.join(base_path, f"{agent_name}_model.pth")
        agent.save_model(filepath)

def load_dqn(env, network_config, agents=["cat", "toy"], base_path="models"):
    """
    Load agent models from files.
    
    Args:
        env: Environment for the agents
        network_config (dict): Configuration for network parameters
        agents (list): List of agent names to load
        base_path (str): Directory to load models from
        
    Returns:
        dict: Dictionary of loaded agents
    """
    # Create agents
    agent_dict = {}
    for agent_name in agents:
        config = network_config.get(agent_name, {})
        agent_dict[agent_name] = DQNAgent(
            agent_name=agent_name,
            env=env,
            network_type=config.get("network_type", "mlp"),
            hidden_dim=config.get("hidden_dim", 64),
            feature_dim=config.get("feature_dim", 64)
        )
        
        filepath = os.path.join(base_path, f"{agent_name}_model.pth")
        if os.path.exists(filepath):
            agent_dict[agent_name].load_model(filepath)
            
    return agent_dict