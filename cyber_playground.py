"""
CyberMARL: Multi-Agent Reinforcement Learning for Cyber Defense
================================================================
Red-Blue Team Adversarial Simulation using Independent Q-Learning

Author: CN Security Course Project
Date: December 2025
Based on: Hierarchical Multi-agent RL for Cyber Network Defense (RLJ 2025)
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import argparse
from collections import defaultdict
import json
import os


# ============================================================================
# GLOBAL CONSTANTS
# ============================================================================

NUM_HOSTS = 3
MAX_STEPS_PER_EPISODE = 50

# Learning Hyperparameters
ALPHA = 0.1      # Learning rate
GAMMA = 0.9      # Discount factor
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

# Reward Values
REWARD_COMPROMISE = 10
REWARD_DETECTION = 5
REWARD_PATCH = 15
REWARD_STEALTH = 1
PENALTY_CAUGHT = -10
PENALTY_COMPROMISED = -10


# ============================================================================
# CYBER ENVIRONMENT CLASS
# ============================================================================

class CyberEnv:
    """
    Simulated cyber network environment with 3 hosts.
    Implements OpenAI Gym-style interface.
    """
    
    def __init__(self):
        # Network topology: 0 <-> 1 <-> 2 (sequential connectivity)
        self.conn_map = {
            0: [1],
            1: [0, 2],
            2: [1]
        }
        
        # Action mappings
        self.red_actions = {
            0: "EXPLOIT_LOCAL",
            1: "MOVE_LATERAL", 
            2: "WAIT"
        }
        
        self.blue_actions = {
            0: "MONITOR_LOCAL",
            1: "QUARANTINE_LOCAL",
            2: "PATCH_ALL"
        }
        
        # State variables (will be initialized in reset())
        self.host_status = {}
        self.red_location = 0
        self.vulnerabilities = {}
        self.blue_alerts = []
        self.patched = False
        self.step_count = 0
        
        # Calculate total state space size
        # 2^3 host states * 3 red positions * 2 patch states = 48 states
        self.state_space_size = (2 ** NUM_HOSTS) * NUM_HOSTS * 2
        
    def reset(self):
        """Initialize environment to starting state"""
        self.host_status = {i: 0 for i in range(NUM_HOSTS)}  # 0=Clean, 1=Compromised
        self.red_location = 0  # Red starts at Host 0
        self.vulnerabilities = {i: True for i in range(NUM_HOSTS)}  # All vulnerable initially
        self.blue_alerts = []
        self.patched = False
        self.step_count = 0
        return self._get_state()
    
    def _get_state(self):
        """
        Encode current network state as a discrete integer.
        State encoding: (host_status_binary, red_location, patched)
        """
        # Convert host status to binary number
        host_binary = sum(self.host_status[i] * (2 ** i) for i in range(NUM_HOSTS))
        
        # Encode as: host_binary * 6 + red_location * 2 + patched
        state = host_binary * (NUM_HOSTS * 2) + self.red_location * 2 + int(self.patched)
        return state
    
    def step(self, red_action, blue_action):
        """
        Execute one timestep: both agents act simultaneously.
        Returns: (new_state, red_reward, blue_reward, done, info)
        """
        self.step_count += 1
        red_reward = 0
        blue_reward = 0
        done = False
        info = {"red_action": self.red_actions[red_action],
                "blue_action": self.blue_actions[blue_action]}
        
        # ========== RED AGENT ACTIONS ==========
        if red_action == 0:  # EXPLOIT_LOCAL
            if self.host_status[self.red_location] == 0:  # Host is clean
                if not self.patched and self.vulnerabilities[self.red_location]:
                    # Successful exploitation
                    self.host_status[self.red_location] = 1
                    red_reward += REWARD_COMPROMISE
                    blue_reward += PENALTY_COMPROMISED
                    info["red_success"] = "Compromised host"
                else:
                    info["red_success"] = "Exploit failed (patched)"
            else:
                info["red_success"] = "Host already compromised"
        
        elif red_action == 1:  # MOVE_LATERAL
            # Move to random adjacent host
            adjacent_hosts = self.conn_map[self.red_location]
            if adjacent_hosts:
                self.red_location = random.choice(adjacent_hosts)
                red_reward += REWARD_STEALTH  # Small reward for persistence
                info["red_success"] = f"Moved to host {self.red_location}"
        
        elif red_action == 2:  # WAIT
            red_reward += REWARD_STEALTH * 0.5  # Very small reward for stealth
            info["red_success"] = "Waiting"
        
        # ========== BLUE AGENT ACTIONS ==========
        if blue_action == 0:  # MONITOR_LOCAL
            # Monitor a random host with detection probability
            monitored_host = random.randint(0, NUM_HOSTS - 1)
            if monitored_host == self.red_location and random.random() < 0.7:
                # Detection successful!
                self.blue_alerts.append(f"Alert: Suspicious activity on Host {monitored_host}")
                blue_reward += REWARD_DETECTION
                red_reward += PENALTY_CAUGHT
                info["blue_success"] = f"Detected Red on Host {monitored_host}"
            else:
                info["blue_success"] = f"Monitored Host {monitored_host} (no detection)"
        
        elif blue_action == 1:  # QUARANTINE_LOCAL
            # Quarantine current Red location (if detected)
            if len(self.blue_alerts) > 0:  # Only works if there are alerts
                if self.host_status[self.red_location] == 1:
                    # Successful quarantine
                    self.host_status[self.red_location] = 0  # Clean the host
                    blue_reward += REWARD_PATCH
                    red_reward += PENALTY_CAUGHT * 2
                    info["blue_success"] = "Quarantined compromised host"
                else:
                    info["blue_success"] = "Quarantine attempted (no effect)"
            else:
                info["blue_success"] = "Quarantine failed (no alerts)"
        
        elif blue_action == 2:  # PATCH_ALL
            if not self.patched:
                # Patch all vulnerabilities
                self.patched = True
                self.vulnerabilities = {i: False for i in range(NUM_HOSTS)}
                blue_reward += REWARD_PATCH
                info["blue_success"] = "Patched all vulnerabilities"
            else:
                info["blue_success"] = "Already patched"
        
        # ========== TERMINATION CONDITIONS ==========
        # Red wins if all hosts compromised
        if all(self.host_status[i] == 1 for i in range(NUM_HOSTS)):
            done = True
            red_reward += 20  # Bonus for total compromise
            blue_reward -= 20
            info["outcome"] = "RED WINS - Total compromise"
        
        # Blue wins if patched and no compromised hosts
        elif self.patched and all(self.host_status[i] == 0 for i in range(NUM_HOSTS)):
            done = True
            blue_reward += 20  # Bonus for successful defense
            red_reward -= 20
            info["outcome"] = "BLUE WINS - Network secured"
        
        # Time limit reached
        elif self.step_count >= MAX_STEPS_PER_EPISODE:
            done = True
            # Evaluate final state
            compromised_count = sum(self.host_status.values())
            if compromised_count > 1:
                info["outcome"] = "RED WINS - Majority compromised"
                red_reward += 10
                blue_reward -= 10
            elif compromised_count == 0:
                info["outcome"] = "BLUE WINS - Network clean"
                blue_reward += 10
                red_reward -= 10
            else:
                info["outcome"] = "DRAW - Stalemate"
        
        new_state = self._get_state()
        return new_state, red_reward, blue_reward, done, info
    
    def render(self):
        """Display current network state"""
        status_symbols = {0: "✓", 1: "✗"}
        print("\n" + "="*50)
        print("NETWORK STATE:")
        for i in range(NUM_HOSTS):
            status = "CLEAN" if self.host_status[i] == 0 else "COMPROMISED"
            symbol = status_symbols[self.host_status[i]]
            red_marker = " 🔴" if i == self.red_location else ""
            vuln = "Patched" if self.patched else "Vulnerable"
            print(f"  Host {i}: {symbol} {status} [{vuln}]{red_marker}")
        print(f"  Alerts: {len(self.blue_alerts)}")
        print("="*50)


# ============================================================================
# Q-LEARNING AGENT CLASS
# ============================================================================

class QLearningAgent:
    """
    Independent Q-Learning Agent for Red or Blue team.
    """
    
    def __init__(self, state_space_size, action_space_size, agent_name="Agent"):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.agent_name = agent_name
        
        # Initialize Q-table
        self.q_table = np.zeros((state_space_size, action_space_size))
        
        # Hyperparameters
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.epsilon = EPSILON_START
        
        # Tracking
        self.total_reward = 0
        self.action_counts = defaultdict(int)
    
    def choose_action(self, state):
        """Epsilon-greedy action selection"""
        if random.random() < self.epsilon:
            # Explore: random action
            action = random.randint(0, self.action_space_size - 1)
        else:
            # Exploit: best known action
            action = np.argmax(self.q_table[state])
        
        self.action_counts[action] += 1
        return action
    
    def learn(self, state, action, reward, next_state, done):
        """Q-learning update rule (Bellman equation)"""
        self.total_reward += reward
        
        # Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        current_q = self.q_table[state, action]
        
        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.q_table[next_state])
            target_q = reward + self.gamma * max_next_q
        
        # Update Q-value
        self.q_table[state, action] = current_q + self.alpha * (target_q - current_q)
    
    def decay_epsilon(self):
        """Reduce exploration rate over time"""
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
    
    def save_qtable(self, filepath):
        """Save Q-table to file"""
        np.save(filepath, self.q_table)
        print(f"[{self.agent_name}] Q-table saved to {filepath}")
    
    def load_qtable(self, filepath):
        """Load Q-table from file"""
        self.q_table = np.load(filepath)
        print(f"[{self.agent_name}] Q-table loaded from {filepath}")


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_agents(num_episodes=1000, verbose=False):
    """
    Train Red and Blue agents through competitive self-play.
    """
    print("\n" + "="*60)
    print("CyberMARL Training Started")
    print("="*60)
    
    # Initialize environment and agents
    env = CyberEnv()
    red_agent = QLearningAgent(env.state_space_size, len(env.red_actions), "RED")
    blue_agent = QLearningAgent(env.state_space_size, len(env.blue_actions), "BLUE")
    
    # Tracking metrics
    red_rewards = []
    blue_rewards = []
    red_wins = []
    blue_wins = []
    epsilon_history = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_red_reward = 0
        episode_blue_reward = 0
        done = False
        
        while not done:
            # Both agents choose actions
            red_action = red_agent.choose_action(state)
            blue_action = blue_agent.choose_action(state)
            
            # Environment step
            next_state, red_reward, blue_reward, done, info = env.step(red_action, blue_action)
            
            # Both agents learn
            red_agent.learn(state, red_action, red_reward, next_state, done)
            blue_agent.learn(state, blue_action, blue_reward, next_state, done)
            
            # Update state
            state = next_state
            episode_red_reward += red_reward
            episode_blue_reward += blue_reward
        
        # Decay exploration
        red_agent.decay_epsilon()
        blue_agent.decay_epsilon()
        
        # Record metrics
        red_rewards.append(episode_red_reward)
        blue_rewards.append(episode_blue_reward)
        epsilon_history.append(red_agent.epsilon)
        
        # Record winner
        if "RED WINS" in info.get("outcome", ""):
            red_wins.append(1)
            blue_wins.append(0)
        elif "BLUE WINS" in info.get("outcome", ""):
            red_wins.append(0)
            blue_wins.append(1)
        else:
            red_wins.append(0)
            blue_wins.append(0)
        
        # Progress reporting
        if (episode + 1) % 100 == 0:
            avg_red = np.mean(red_rewards[-100:])
            avg_blue = np.mean(blue_rewards[-100:])
            red_win_rate = np.mean(red_wins[-100:]) * 100
            blue_win_rate = np.mean(blue_wins[-100:]) * 100
            
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Red Avg: {avg_red:.1f} | Blue Avg: {avg_blue:.1f} | "
                  f"Red Wins: {red_win_rate:.0f}% | Blue Wins: {blue_win_rate:.0f}% | "
                  f"ε={red_agent.epsilon:.3f}")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    
    return env, red_agent, blue_agent, {
        'red_rewards': red_rewards,
        'blue_rewards': blue_rewards,
        'red_wins': red_wins,
        'blue_wins': blue_wins,
        'epsilon_history': epsilon_history
    }


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_training_results(metrics, save_path='results/plots'):
    """Generate comprehensive training visualizations"""
    os.makedirs(save_path, exist_ok=True)
    
    episodes = range(1, len(metrics['red_rewards']) + 1)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('CyberMARL Training Results', fontsize=16, fontweight='bold')
    
    # 1. Cumulative Rewards
    ax1 = axes[0, 0]
    window = 50
    red_ma = np.convolve(metrics['red_rewards'], np.ones(window)/window, mode='valid')
    blue_ma = np.convolve(metrics['blue_rewards'], np.ones(window)/window, mode='valid')
    
    ax1.plot(episodes, metrics['red_rewards'], 'r-', alpha=0.3, linewidth=0.5)
    ax1.plot(range(window, len(episodes) + 1), red_ma, 'r-', linewidth=2, label='Red Agent')
    ax1.plot(episodes, metrics['blue_rewards'], 'b-', alpha=0.3, linewidth=0.5)
    ax1.plot(range(window, len(episodes) + 1), blue_ma, 'b-', linewidth=2, label='Blue Agent')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Learning Curves (50-episode moving average)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    
    # 2. Win Rates
    ax2 = axes[0, 1]
    window = 100
    red_wr = np.convolve(metrics['red_wins'], np.ones(window)/window, mode='valid') * 100
    blue_wr = np.convolve(metrics['blue_wins'], np.ones(window)/window, mode='valid') * 100
    
    ax2.plot(range(window, len(episodes) + 1), red_wr, 'r-', linewidth=2, label='Red Win Rate')
    ax2.plot(range(window, len(episodes) + 1), blue_wr, 'b-', linewidth=2, label='Blue Win Rate')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Win Rate (%)')
    ax2.set_title('Win Rates (100-episode moving average)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])
    
    # 3. Epsilon Decay
    ax3 = axes[1, 0]
    ax3.plot(episodes, metrics['epsilon_history'], 'g-', linewidth=2)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Epsilon')
    ax3.set_title('Exploration Rate Decay')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1.05])
    
    # 4. Reward Distribution
    ax4 = axes[1, 1]
    ax4.hist(metrics['red_rewards'], bins=30, alpha=0.6, color='red', label='Red')
    ax4.hist(metrics['blue_rewards'], bins=30, alpha=0.6, color='blue', label='Blue')
    ax4.set_xlabel('Episode Reward')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Reward Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/training_results.png', dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to {save_path}/training_results.png")
    plt.show()


# ============================================================================
# DEMO FUNCTION
# ============================================================================

def run_demo(env, red_agent, blue_agent, render=True):
    """Run a single episode with trained agents"""
    print("\n" + "="*60)
    print("DEMO: Trained Agents in Action")
    print("="*60)
    
    state = env.reset()
    done = False
    step = 0
    total_red_reward = 0
    total_blue_reward = 0
    
    # Use greedy policy (no exploration)
    red_agent.epsilon = 0
    blue_agent.epsilon = 0
    
    if render:
        env.render()
    
    while not done:
        step += 1
        
        # Choose actions
        red_action = red_agent.choose_action(state)
        blue_action = blue_agent.choose_action(state)
        
        # Step environment
        next_state, red_reward, blue_reward, done, info = env.step(red_action, blue_action)
        
        total_red_reward += red_reward
        total_blue_reward += blue_reward
        
        # Display step info
        print(f"\n--- Step {step} ---")
        print(f"Red Action: {env.red_actions[red_action]} → {info['red_success']}")
        print(f"Blue Action: {env.blue_actions[blue_action]} → {info['blue_success']}")
        print(f"Rewards: Red={red_reward:+.0f}, Blue={blue_reward:+.0f}")
        
        if render:
            env.render()
        
        state = next_state
        
        if step >= MAX_STEPS_PER_EPISODE:
            break
    
    print("\n" + "="*60)
    print(f"FINAL OUTCOME: {info.get('outcome', 'UNKNOWN')}")
    print(f"Total Rewards: Red={total_red_reward:+.0f}, Blue={total_blue_reward:+.0f}")
    print("="*60)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='CyberMARL - Multi-Agent RL for Cyber Defense')
    parser.add_argument('--train', action='store_true', help='Train agents')
    parser.add_argument('--demo', action='store_true', help='Run demo with trained agents')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    
    args = parser.parse_args()
    
    if args.train:
        # Train agents
        env, red_agent, blue_agent, metrics = train_agents(num_episodes=args.episodes)
        
        # Save Q-tables
        os.makedirs('results/trained_models', exist_ok=True)
        red_agent.save_qtable('results/trained_models/red_agent_qtable.npy')
        blue_agent.save_qtable('results/trained_models/blue_agent_qtable.npy')
        
        # Save metrics
        with open('results/logs/training_metrics.json', 'w') as f:
            json.dump({k: [float(v) for v in vals] for k, vals in metrics.items()}, f)
        
        # Visualize
        if args.visualize:
            plot_training_results(metrics)
        
        # Run demo
        if args.demo:
            run_demo(env, red_agent, blue_agent)
    
    elif args.demo:
        # Load trained agents
        env = CyberEnv()
        red_agent = QLearningAgent(env.state_space_size, len(env.red_actions), "RED")
        blue_agent = QLearningAgent(env.state_space_size, len(env.blue_actions), "BLUE")
        
        try:
            red_agent.load_qtable('results/trained_models/red_agent_qtable.npy')
            blue_agent.load_qtable('results/trained_models/blue_agent_qtable.npy')
            run_demo(env, red_agent, blue_agent)
        except FileNotFoundError:
            print("Error: Trained models not found. Please train first using --train")
    
    elif args.visualize:
        # Load and visualize metrics
        try:
            with open('results/logs/training_metrics.json', 'r') as f:
                metrics = json.load(f)
            plot_training_results(metrics)
        except FileNotFoundError:
            print("Error: Training metrics not found. Please train first using --train")
    
    else:
        print("CyberMARL - Multi-Agent Reinforcement Learning for Cyber Defense")
        print("\nUsage:")
        print("  python cyber_playground.py --train --episodes 1000 --visualize")
        print("  python cyber_playground.py --demo")
        print("  python cyber_playground.py --visualize")
        print("\nRun with -h for more options")


if __name__ == "__main__":
    main()