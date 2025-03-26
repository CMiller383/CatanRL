import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import deque

from .agent import PPOAgent
from .environment import CatanEnvironment
from agent.base import AgentType


def parse_args():
    parser = argparse.ArgumentParser(description='Train a PPO agent for Settlers of Catan')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes to train')
    parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--hidden-size', type=int, default=256, help='Hidden layer size')
    parser.add_argument('--save-dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--log-interval', type=int, default=10, help='Log interval')
    parser.add_argument('--save-interval', type=int, default=100, help='Save interval')
    parser.add_argument('--eval-interval', type=int, default=50, help='Evaluation interval')
    parser.add_argument('--num-eval-episodes', type=int, default=10, help='Number of evaluation episodes')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    return parser.parse_args()


def evaluate_agent(agent, env, num_episodes=10):
    """Evaluate the agent without training"""
    rewards = []
    victory_points = []
    
    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Select best action
            action, _, _ = agent.select_action(state, deterministic=True)
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            state = next_state
        
        rewards.append(episode_reward)
        victory_points.append(info.get('victory_points', 0))
    
    return np.mean(rewards), np.mean(victory_points)


def visualize_training(episode_rewards, victory_points, moving_avg_rewards, save_path=None):
    """Visualize training progress"""
    plt.figure(figsize=(12, 10))
    
    # Plot episode rewards
    plt.subplot(2, 1, 1)
    plt.plot(episode_rewards)
    plt.plot(moving_avg_rewards)
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend(['Episode Reward', 'Moving Average'])
    
    # Plot victory points
    plt.subplot(2, 1, 2)
    plt.plot(victory_points)
    plt.title('Victory Points')
    plt.xlabel('Episode')
    plt.ylabel('Victory Points')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def train_agent():
    args = parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create directory for saving models
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Create agent and environment
    agent = PPOAgent(
        player_id=1,
        hidden_size=args.hidden_size,
        learning_rate=args.lr
    )
    env = CatanEnvironment()
    
    # Training statistics
    episode_rewards = []
    victory_points = []
    running_reward = deque(maxlen=100)
    
    # Training loop
    for episode in tqdm(range(args.episodes)):
        state = env.reset()
        episode_reward = 0
        done = False
        
        # Play one episode
        while not done:
            # Select action
            action, action_prob, value = agent.select_action(state)
            
            # Take action in environment
            next_state, reward, done, info = env.step(action)
            
            # Store in memory
            agent.memory.add(state, action, reward, next_state, action_prob, value, done)
            
            # Move to next state
            state = next_state
            episode_reward += reward
            
            # Update if memory is full
            if len(agent.memory.states) >= agent.batch_size:
                agent._update()
                agent.memory.clear()
        
        # Final update for episode
        if len(agent.memory.states) > 0:
            agent._update()
            agent.memory.clear()
        
        # Store statistics
        episode_rewards.append(episode_reward)
        victory_points.append(info.get('victory_points', 0))
        running_reward.append(episode_reward)
        
        # Log progress
        if (episode + 1) % args.log_interval == 0:
            avg_reward = np.mean(running_reward)
            print(f"Episode {episode+1}/{args.episodes}, Reward: {episode_reward:.2f}, "
                  f"Average Reward: {avg_reward:.2f}, VP: {info.get('victory_points', 0)}")
        
        # Save model
        if (episode + 1) % args.save_interval == 0:
            save_path = os.path.join(args.save_dir, f"ppo_agent_ep{episode+1}.pt")
            agent.save(save_path)
            print(f"Model saved to {save_path}")
        
        # Evaluate agent
        if (episode + 1) % args.eval_interval == 0:
            eval_reward, eval_vp = evaluate_agent(agent, env, args.num_eval_episodes)
            print(f"Evaluation - Reward: {eval_reward:.2f}, Victory Points: {eval_vp:.2f}")
    
    # Calculate moving average rewards
    window_size = min(100, args.episodes)
    moving_avg_rewards = []
    for i in range(len(episode_rewards)):
        if i < window_size - 1:
            moving_avg_rewards.append(np.mean(episode_rewards[:i+1]))
        else:
            moving_avg_rewards.append(np.mean(episode_rewards[i-window_size+1:i+1]))
    
    # Final model save
    final_path = os.path.join(args.save_dir, "ppo_agent_final.pt")
    agent.save(final_path)
    print(f"Final model saved to {final_path}")
    
    # Visualize training progress
    visualize_training(
        episode_rewards, 
        victory_points, 
        moving_avg_rewards,
        save_path=os.path.join(args.save_dir, "training_progress.png")
    )


if __name__ == "__main__":
    train_agent()