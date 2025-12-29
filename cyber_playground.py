"""
CyberMARL: Multi-Agent Reinforcement Learning for Cyber Defense
================================================================
Red-Blue Team Adversarial Simulation using Independent Q-Learning
Updated with: NetworkX Visualization, Colorama, and Zero-Day Exploits

Author: CN Security Course Project
Date: December 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
import argparse
from collections import defaultdict
import json
import os
import time

# Import Colorama for terminal styling
try:
    from colorama import init, Fore, Style, Back
    init(autoreset=True)
except ImportError:
    class ColorMock:
        def __getattr__(self, name): return ""
    Fore = Style = Back = ColorMock()

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
REWARD_COMPROMISE = 20   # Increased from 10 (Make exploiting worth the risk)
REWARD_DETECTION = 5
REWARD_PATCH = 15
REWARD_STEALTH = 0       # Set to 0 (Moving should be a means to an end, not a goal)
PENALTY_CAUGHT = -10
PENALTY_COMPROMISED = -10
COST_PATCHING_OP = -2

# Buff Zero-Day slightly to make it more tempting
ZERO_DAY_PROB = 0.12     # Increase to 25%


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
        
        self.red_actions = {0: "EXPLOIT_LOCAL", 1: "MOVE_LATERAL", 2: "WAIT"}
        self.blue_actions = {0: "MONITOR_LOCAL", 1: "QUARANTINE_LOCAL", 2: "PATCH_ALL"}
        
        self.reset()
        self.state_space_size = (2 ** NUM_HOSTS) * NUM_HOSTS * 2
        
    def reset(self):
        self.host_status = {i: 0 for i in range(NUM_HOSTS)}  # 0=Clean, 1=Compromised
        self.red_location = 0
        self.vulnerabilities = {i: True for i in range(NUM_HOSTS)}
        self.blue_alerts = []
        self.patched = False
        self.step_count = 0
        return self._get_state()
    
    def _get_state(self):
        host_binary = sum(self.host_status[i] * (2 ** i) for i in range(NUM_HOSTS))
        state = host_binary * (NUM_HOSTS * 2) + self.red_location * 2 + int(self.patched)
        return state
    
    def step(self, red_action, blue_action):
        self.step_count += 1
        red_reward = 0
        blue_reward = 0
        done = False
        info = {"red_action": self.red_actions[red_action],
                "blue_action": self.blue_actions[blue_action]}
        
        # ========== RED AGENT ACTIONS ==========
        if red_action == 0:  # EXPLOIT_LOCAL
            if self.host_status[self.red_location] == 0:
                is_vuln = self.vulnerabilities[self.red_location]
                # Roll for Zero-Day exploit (works even if patched)
                is_zero_day = random.random() < ZERO_DAY_PROB
                
                if is_vuln or is_zero_day:
                    self.host_status[self.red_location] = 1
                    red_reward += REWARD_COMPROMISE
                    blue_reward += PENALTY_COMPROMISED
                    
                    if not is_vuln and is_zero_day:
                        info["red_success"] = "used ZERO-DAY EXPLOIT!"
                        red_reward += 5 # Bonus for successful zero-day
                    else:
                        info["red_success"] = "Compromised host"
                else:
                    info["red_success"] = "Exploit failed (patched)"
                    red_reward -= 1 # Small waste of time penalty
            else:
                info["red_success"] = "Host already compromised"
                red_reward -= 1
        
        elif red_action == 1:  # MOVE_LATERAL
            adjacent_hosts = self.conn_map[self.red_location]
            if adjacent_hosts:
                self.red_location = random.choice(adjacent_hosts)
                red_reward += REWARD_STEALTH
                info["red_success"] = f"Moved to host {self.red_location}"
        
        elif red_action == 2:  # WAIT
            red_reward += REWARD_STEALTH * 0.5
            info["red_success"] = "Waiting"
        
        # ========== BLUE AGENT ACTIONS ==========
        if blue_action == 0:  # MONITOR_LOCAL
            monitored_host = random.randint(0, NUM_HOSTS - 1)
            if monitored_host == self.red_location and random.random() < 0.7:
                self.blue_alerts.append(f"Alert: Suspicious activity on Host {monitored_host}")
                blue_reward += REWARD_DETECTION
                red_reward += PENALTY_CAUGHT
                info["blue_success"] = f"Detected Red on Host {monitored_host}"
            else:
                info["blue_success"] = f"Monitored Host {monitored_host} (no detection)"
        
        elif blue_action == 1:  # QUARANTINE_LOCAL
            if len(self.blue_alerts) > 0:
                if self.host_status[self.red_location] == 1:
                    self.host_status[self.red_location] = 0
                    blue_reward += REWARD_PATCH
                    red_reward += PENALTY_CAUGHT * 2
                    info["blue_success"] = "Quarantined compromised host"
                else:
                    info["blue_success"] = "Quarantine attempted (no effect)"
                    blue_reward -= 2 # False positive cost
            else:
                info["blue_success"] = "Quarantine failed (no alerts)"
                blue_reward -= 1
        
        elif blue_action == 2:  # PATCH_ALL
            if not self.patched:
                self.patched = True
                self.vulnerabilities = {i: False for i in range(NUM_HOSTS)}
                blue_reward += COST_PATCHING_OP # Operational cost
                blue_reward += REWARD_PATCH
                info["blue_success"] = "Patched all vulnerabilities"
            else:
                blue_reward -= 2 # Penalty for redundant action
                info["blue_success"] = "Already patched"
        
        # ========== TERMINATION CONDITIONS ==========
        if all(self.host_status[i] == 1 for i in range(NUM_HOSTS)):
            done = True
            red_reward += 20
            blue_reward -= 20
            info["outcome"] = "RED WINS - Total compromise"
        
        elif self.patched and all(self.host_status[i] == 0 for i in range(NUM_HOSTS)) and self.step_count > 40:
            # Blue only wins early if Red is completely shut out for a long time
            done = True
            blue_reward += 20
            red_reward -= 20
            info["outcome"] = "BLUE WINS - Network secured"
        
        elif self.step_count >= MAX_STEPS_PER_EPISODE:
            done = True
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
        status_symbols = {0: f"{Fore.GREEN}✓ SECURE{Style.RESET_ALL}", 
                          1: f"{Fore.RED}☠ COMPROMISED{Style.RESET_ALL}"}
        
        print("\n" + f"{Fore.CYAN}="*50)
        print(f"{Style.BRIGHT}NETWORK STATE:")
        
        for i in range(NUM_HOSTS):
            status = status_symbols[self.host_status[i]]
            if i == self.red_location:
                host_str = f"{Back.RED}{Fore.WHITE} HOST {i} {Style.RESET_ALL}"
                marker = f" <--- {Fore.RED}{Style.BRIGHT}RED AGENT HERE{Style.RESET_ALL}"
            else:
                host_str = f" Host {i}"
                marker = ""
                
            vuln = f"{Fore.GREEN}Patched{Style.RESET_ALL}" if self.patched else f"{Fore.YELLOW}Vulnerable{Style.RESET_ALL}"
            print(f" {host_str}: {status} [{vuln}]{marker}")
            
        print(f" Alerts: {Fore.YELLOW}{len(self.blue_alerts)}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}="*50 + Style.RESET_ALL)


# ============================================================================
# Q-LEARNING AGENT CLASS
# ============================================================================

class QLearningAgent:
    def __init__(self, state_space_size, action_space_size, agent_name="Agent"):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.agent_name = agent_name
        self.q_table = np.zeros((state_space_size, action_space_size))
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.epsilon = EPSILON_START
        self.total_reward = 0
    
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space_size - 1)
        else:
            return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state, done):
        self.total_reward += reward
        current_q = self.q_table[state, action]
        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.q_table[next_state])
            target_q = reward + self.gamma * max_next_q
        self.q_table[state, action] = current_q + self.alpha * (target_q - current_q)
    
    def decay_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
    
    def save_qtable(self, filepath):
        np.save(filepath, self.q_table)
        print(f"[{self.agent_name}] Q-table saved to {filepath}")
    
    def load_qtable(self, filepath):
        self.q_table = np.load(filepath)
        print(f"[{self.agent_name}] Q-table loaded from {filepath}")


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_agents(num_episodes=1000, verbose=False):
    print("\n" + "="*60)
    print("CyberMARL Training Started")
    print("="*60)
    
    env = CyberEnv()
    red_agent = QLearningAgent(env.state_space_size, len(env.red_actions), "RED")
    blue_agent = QLearningAgent(env.state_space_size, len(env.blue_actions), "BLUE")
    
    red_rewards, blue_rewards = [], []
    red_wins, blue_wins = [], []
    
    for episode in range(num_episodes):
        state = env.reset()
        e_red_r, e_blue_r = 0, 0
        done = False
        
        while not done:
            red_action = red_agent.choose_action(state)
            blue_action = blue_agent.choose_action(state)
            next_state, r_rew, b_rew, done, info = env.step(red_action, blue_action)
            red_agent.learn(state, red_action, r_rew, next_state, done)
            blue_agent.learn(state, blue_action, b_rew, next_state, done)
            state = next_state
            e_red_r += r_rew
            e_blue_r += b_rew
        
        red_agent.decay_epsilon()
        blue_agent.decay_epsilon()
        red_rewards.append(e_red_r)
        blue_rewards.append(e_blue_r)
        
        if "RED WINS" in info.get("outcome", ""):
            red_wins.append(1); blue_wins.append(0)
        elif "BLUE WINS" in info.get("outcome", ""):
            red_wins.append(0); blue_wins.append(1)
        else:
            red_wins.append(0); blue_wins.append(0)
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes} | Red Avg: {np.mean(red_rewards[-100:]):.1f} | Blue Avg: {np.mean(blue_rewards[-100:]):.1f} | ε={red_agent.epsilon:.3f}")
    
    return env, red_agent, blue_agent, {'red_rewards': red_rewards, 'blue_rewards': blue_rewards}


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def update_network_plot(env, ax, step_num):
    ax.clear()
    G = nx.Graph()
    G.add_edges_from([(0,1), (1,2)])
    pos = {0: (0, 0), 1: (1, 0), 2: (2, 0)}
    
    node_colors = ['#ff4d4d' if env.host_status[i] == 1 else '#4dff88' for i in range(NUM_HOSTS)]
    
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=2000, edgecolors='black')
    nx.draw_networkx_edges(G, pos, ax=ax, width=2)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=12, font_weight='bold')
    
    # Red Agent Highlight (Thick red ring)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=[env.red_location], 
                           node_size=2800, node_color='none', edgecolors='red', linewidths=3)
    
    ax.set_title(f"Network Topology - Step {step_num}", fontsize=15)
    ax.text(1, -0.3, "Green = Secure | Red = Compromised | Ring = Red Agent", fontsize=10, ha='center')
    ax.set_xlim(-0.5, 2.5); ax.set_ylim(-0.5, 0.5); ax.axis('off')

def plot_training_results(metrics, save_path='results/plots'):
    os.makedirs(save_path, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(metrics['red_rewards'], label='Red Reward', alpha=0.6, color='red')
    plt.plot(metrics['blue_rewards'], label='Blue Reward', alpha=0.6, color='blue')
    plt.title('Training Rewards (Zero-Day Enabled)')
    plt.legend()
    plt.savefig(f'{save_path}/training_results.png')
    plt.close()


# ============================================================================
# DEMO EXECUTION
# ============================================================================

def run_demo(env, red_agent, blue_agent, render=True):
    print("\n" + "="*60)
    print(f"{Fore.MAGENTA}{Style.BRIGHT}DEMO: Zero-Day Enabled Simulation{Style.RESET_ALL}")
    print("="*60)
    
    state = env.reset()
    done = False
    step = 0
    red_agent.epsilon = 0; blue_agent.epsilon = 0
    
    if render:
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 5))
        update_network_plot(env, ax, step)
        env.render()
        plt.pause(1)

    while not done:
        step += 1
        red_action = red_agent.choose_action(state)
        blue_action = blue_agent.choose_action(state)
        next_state, r_rew, b_rew, done, info = env.step(red_action, blue_action)
        
        print(f"\n{Style.BRIGHT}--- Step {step} ---{Style.RESET_ALL}")
        
        # Red Action Coloring
        r_msg = info['red_success']
        if "ZERO-DAY" in r_msg:
            r_color = Back.RED + Fore.WHITE + Style.BRIGHT
        elif "Compromised" in r_msg or "Moved" in r_msg:
            r_color = Fore.RED
        else:
            r_color = Fore.YELLOW
        print(f"{Fore.RED}Red Action:{Style.RESET_ALL} {env.red_actions[red_action]} → {r_color}{r_msg}{Style.RESET_ALL}")
        
        # Blue Action Coloring
        b_msg = info['blue_success']
        b_color = Fore.BLUE if "Detected" in b_msg or "Quarantined" in b_msg else Fore.CYAN
        print(f"{Fore.BLUE}Blue Action:{Style.RESET_ALL} {env.blue_actions[blue_action]} → {b_color}{b_msg}{Style.RESET_ALL}")
        
        print(f"Rewards: Red={r_rew:+.0f}, Blue={b_rew:+.0f}")
        
        if render:
            env.render()
            update_network_plot(env, ax, step)
            plt.draw()
            plt.pause(1.5)
        
        state = next_state
        if step >= MAX_STEPS_PER_EPISODE: break
    
    if render:
        plt.ioff(); plt.show()

    print("\n" + "="*60)
    outcome = info.get('outcome', 'UNKNOWN')
    print(f"FINAL OUTCOME: {Style.BRIGHT}{outcome}{Style.RESET_ALL}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()
    
    if args.train:
        env, red, blue, metrics = train_agents(args.episodes)
        os.makedirs('results/trained_models', exist_ok=True)
        red.save_qtable('results/trained_models/red_agent_qtable.npy')
        blue.save_qtable('results/trained_models/blue_agent_qtable.npy')
        if args.visualize: plot_training_results(metrics)
        if args.demo: run_demo(env, red, blue)
    elif args.demo:
        env = CyberEnv()
        red = QLearningAgent(env.state_space_size, len(env.red_actions), "RED")
        blue = QLearningAgent(env.state_space_size, len(env.blue_actions), "BLUE")
        try:
            red.load_qtable('results/trained_models/red_agent_qtable.npy')
            blue.load_qtable('results/trained_models/blue_agent_qtable.npy')
            run_demo(env, red, blue)
        except:
            print("Train first!")

if __name__ == "__main__":
    main()