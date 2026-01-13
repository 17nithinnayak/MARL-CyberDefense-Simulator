"""
Baseline Comparison Agents for CyberMARL
Demonstrates your Q-Learning agents are actually learning
"""

import random
import numpy as np

class RandomAgent:
    """
    Baseline 1: Pure Random Agent
    Always picks random actions (no learning)
    """
    def __init__(self, action_space_size, agent_name="Random"):
        self.action_space_size = action_space_size
        self.agent_name = agent_name
    
    def choose_action(self, state):
        return random.randint(0, self.action_space_size - 1)
    
    def learn(self, state, action, reward, next_state, done):
        pass  # Random agents don't learn


class GreedyAgent:
    """
    Baseline 2: Rule-Based Greedy Agent
    Uses simple heuristics (no learning)
    """
    def __init__(self, agent_type, agent_name="Greedy"):
        self.agent_type = agent_type  # 'red' or 'blue'
        self.agent_name = agent_name
    
    def choose_action(self, state):
        if self.agent_type == 'red':
            # Red: Always try to exploit
            return 0  # EXPLOIT_LOCAL
        else:
            # Blue: Always try to patch
            return 2  # PATCH_ALL
    
    def learn(self, state, action, reward, next_state, done):
        pass  # Greedy agents don't learn


class AlwaysExploitRed:
    """
    Baseline 3: Red agent that only exploits
    """
    def __init__(self, agent_name="AlwaysExploit"):
        self.agent_name = agent_name
    
    def choose_action(self, state):
        return 0  # Always EXPLOIT
    
    def learn(self, state, action, reward, next_state, done):
        pass


class AlwaysPatchBlue:
    """
    Baseline 4: Blue agent that only patches
    """
    def __init__(self, agent_name="AlwaysPatch"):
        self.agent_name = agent_name
    
    def choose_action(self, state):
        return 2  # Always PATCH
    
    def learn(self, state, action, reward, next_state, done):
        pass


def run_baseline_comparison(env, num_episodes=100):
    """
    Compare all baselines against each other and Q-Learning
    """
    results = {}
    
    # Test configurations
    configs = [
        ("Random vs Random", RandomAgent(3, "Red-Random"), RandomAgent(3, "Blue-Random")),
        ("Greedy vs Greedy", GreedyAgent('red', "Red-Greedy"), GreedyAgent('blue', "Blue-Greedy")),
        ("AlwaysExploit vs AlwaysPatch", AlwaysExploitRed(), AlwaysPatchBlue()),
    ]
    
    for name, red_agent, blue_agent in configs:
        total_red_reward = 0
        total_blue_reward = 0
        red_wins = 0
        blue_wins = 0
        
        for episode in range(num_episodes):
            state = env.reset()
            episode_red_reward = 0
            episode_blue_reward = 0
            done = False
            
            while not done:
                red_action = red_agent.choose_action(state)
                blue_action = blue_agent.choose_action(state)
                next_state, red_reward, blue_reward, done, info = env.step(red_action, blue_action)
                
                episode_red_reward += red_reward
                episode_blue_reward += blue_reward
                state = next_state
            
            total_red_reward += episode_red_reward
            total_blue_reward += episode_blue_reward
            
            if "RED WINS" in info.get("outcome", ""):
                red_wins += 1
            elif "BLUE WINS" in info.get("outcome", ""):
                blue_wins += 1
        
        results[name] = {
            'red_avg_reward': total_red_reward / num_episodes,
            'blue_avg_reward': total_blue_reward / num_episodes,
            'red_win_rate': red_wins / num_episodes,
            'blue_win_rate': blue_wins / num_episodes
        }
    
    return results


def print_comparison_table(baseline_results, qlearning_results):
    """
    Print nice comparison table
    """
    print("\n" + "="*80)
    print("BASELINE COMPARISON RESULTS")
    print("="*80)
    print(f"{'Agent Type':<30} | {'Red Reward':>12} | {'Blue Reward':>12} | {'Red Win %':>10}")
    print("-"*80)
    
    for name, results in baseline_results.items():
        print(f"{name:<30} | {results['red_avg_reward']:>12.1f} | "
              f"{results['blue_avg_reward']:>12.1f} | {results['red_win_rate']*100:>9.1f}%")
    
    print("-"*80)
    print(f"{'Q-Learning (Ours)':<30} | {qlearning_results['red_avg_reward']:>12.1f} | "
          f"{qlearning_results['blue_avg_reward']:>12.1f} | {qlearning_results['red_win_rate']*100:>9.1f}%")
    print("="*80)
    
    # Calculate improvement
    random_baseline = baseline_results['Random vs Random']
    improvement = ((qlearning_results['red_avg_reward'] - random_baseline['red_avg_reward']) / 
                   abs(random_baseline['red_avg_reward'])) * 100
    
    print(f"\n✓ Q-Learning improvement over random: {improvement:+.1f}%")
    print("="*80)


# Usage example
if __name__ == "__main__":
    from cyber_playground import CyberEnv
    
    env = CyberEnv()
    
    # Run baseline comparisons
    print("Running baseline comparisons (100 episodes each)...")
    baseline_results = run_baseline_comparison(env, num_episodes=100)
    
    # Add your Q-Learning results here
    qlearning_results = {
        'red_avg_reward': 15.1,  # From your final training
        'blue_avg_reward': -28.5,
        'red_win_rate': 0.60
    }
    
    print_comparison_table(baseline_results, qlearning_results)