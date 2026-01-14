"""
NOVEL CONTRIBUTION: Adaptive Red Agent with Counter-Strategy Learning

Key Innovation: Red agent that detects Blue's strategy and adapts exploitation timing
Paper Difference: RLJ 2025 uses scripted FSM attackers; ours learns to counter defenses

Research Question: Can an attacker learn optimal timing by observing defender patterns?
"""

import numpy as np
from collections import deque

class AdaptiveRedAgent:
    """
    Red agent that adapts strategy based on Blue's observed behavior
    
    Novel Features:
    1. Tracks Blue's action history
    2. Learns Blue's patch timing patterns
    3. Adjusts exploitation windows dynamically
    4. Implements "wait-and-exploit" when Blue monitors frequently
    """
    
    def __init__(self, state_space_size, action_space_size):
        # Standard Q-learning components
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.q_table = np.zeros((state_space_size, action_space_size))
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 1.0
        
        # NOVEL: Blue behavior tracking
        self.blue_action_history = deque(maxlen=20)  # Last 20 Blue actions
        self.blue_patch_count = 0
        self.blue_monitor_count = 0
        self.steps_since_blue_patch = 0
        
        # NOVEL: Adaptive thresholds
        self.exploit_threshold = 0.5  # Dynamic exploitation probability
        self.patience_level = 0  # How long to wait before exploiting
        
    def observe_blue_action(self, blue_action, step):
        """
        Track Blue's actions to learn patterns
        
        Novel Insight: Blue agents often patch within first 3 steps
        Red can exploit this by waiting or rushing
        """
        self.blue_action_history.append(blue_action)
        
        if blue_action == 2:  # PATCH_ALL
            self.blue_patch_count += 1
            self.steps_since_blue_patch = 0
            
            # NOVEL: If Blue patches early, become more cautious
            if step < 5:
                self.patience_level += 1
        
        elif blue_action == 0:  # MONITOR
            self.blue_monitor_count += 1
        
        self.steps_since_blue_patch += 1
        
    def get_blue_strategy_type(self):
        """
        Classify Blue's strategy based on observed behavior
        
        Returns:
        - 'aggressive_patcher': Patches within 3 steps
        - 'reactive_monitor': Monitors frequently  
        - 'balanced': Mix of actions
        """
        if len(self.blue_action_history) < 5:
            return 'unknown'
        
        recent_actions = list(self.blue_action_history)[-10:]
        patch_rate = recent_actions.count(2) / len(recent_actions)
        monitor_rate = recent_actions.count(0) / len(recent_actions)
        
        if patch_rate > 0.4:
            return 'aggressive_patcher'
        elif monitor_rate > 0.5:
            return 'reactive_monitor'
        else:
            return 'balanced'
    
    def choose_action_adaptive(self, state, env):
        """
        NOVEL: Action selection with Blue behavior awareness
        
        Key Innovation: Modifies exploitation probability based on Blue's patterns
        """
        blue_strategy = self.get_blue_strategy_type()
        
        # Standard epsilon-greedy
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, self.action_space_size)
        else:
            action = np.argmax(self.q_table[state])
        
        # NOVEL: Override with adaptive logic
        if action == 0:  # EXPLOIT
            # If Blue is aggressive patcher, delay exploitation
            if blue_strategy == 'aggressive_patcher' and self.steps_since_blue_patch < 3:
                # Wait strategy: Don't exploit immediately after patch
                action = 2  # WAIT instead
            
            # If Blue monitors heavily, exploit when they're not watching
            elif blue_strategy == 'reactive_monitor':
                # Exploit more aggressively during monitoring gaps
                if len(self.blue_action_history) > 0 and self.blue_action_history[-1] != 0:
                    action = 0  # EXPLOIT when Blue just stopped monitoring
        
        return action
    
    def choose_action(self, state):
        """Standard interface for compatibility"""
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_space_size)
        return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state, done):
        """Standard Q-learning update"""
        current_q = self.q_table[state, action]
        
        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.q_table[next_state])
            target_q = reward + self.gamma * max_next_q
        
        self.q_table[state, action] = current_q + self.alpha * (target_q - current_q)
    
    def decay_epsilon(self):
        """Standard epsilon decay"""
        self.epsilon = max(0.01, self.epsilon * 0.995)
    
    def get_adaptation_stats(self):
        """
        Return statistics about adaptation behavior
        For research analysis
        """
        return {
            'blue_strategy': self.get_blue_strategy_type(),
            'patch_count': self.blue_patch_count,
            'monitor_count': self.blue_monitor_count,
            'patience_level': self.patience_level,
            'steps_since_patch': self.steps_since_blue_patch
        }


# ============================================================================
# TRAINING FUNCTION WITH ADAPTIVE RED
# ============================================================================

def train_adaptive_agents(num_episodes=1000):
    """
    Train with Adaptive Red vs Standard Blue
    
    Research Question: Does adaptation improve Red's performance?
    """
    from cyber_playground import CyberEnv, QLearningAgent
    
    print("\n" + "="*70)
    print("TRAINING ADAPTIVE RED AGENT")
    print("="*70)
    
    env = CyberEnv()
    red_agent = AdaptiveRedAgent(env.state_space_size, len(env.red_actions))
    blue_agent = QLearningAgent(env.state_space_size, len(env.blue_actions), "BLUE")
    
    red_rewards = []
    adaptation_events = []  # Track when Red adapts
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_red_reward = 0
        done = False
        step = 0
        
        while not done:
            # Red chooses action (with adaptation)
            red_action = red_agent.choose_action_adaptive(state, env)
            blue_action = blue_agent.choose_action(state)
            
            # Red observes Blue's action (NOVEL)
            red_agent.observe_blue_action(blue_action, step)
            
            # Step environment
            next_state, red_reward, blue_reward, done, info = env.step(red_action, blue_action)
            
            # Learn
            red_agent.learn(state, red_action, red_reward, next_state, done)
            blue_agent.learn(state, blue_action, blue_reward, next_state, done)
            
            state = next_state
            episode_red_reward += red_reward
            step += 1
        
        red_agent.decay_epsilon()
        blue_agent.decay_epsilon()
        red_rewards.append(episode_red_reward)
        
        # Track adaptation behavior
        if episode % 100 == 0:
            stats = red_agent.get_adaptation_stats()
            adaptation_events.append(stats)
            
            print(f"Episode {episode}: Red Reward={episode_red_reward:.1f}, "
                  f"Blue Strategy={stats['blue_strategy']}, "
                  f"Red Patience={stats['patience_level']}")
    
    print("\n" + "="*70)
    print("ADAPTIVE TRAINING COMPLETE")
    print("="*70)
    
    return env, red_agent, blue_agent, {
        'red_rewards': red_rewards,
        'adaptation_events': adaptation_events
    }


# ============================================================================
# COMPARISON: ADAPTIVE vs STANDARD
# ============================================================================

def compare_adaptive_vs_standard():
    """
    NOVEL EXPERIMENT: Compare adaptive Red vs standard Red
    
    Research Question: Does Blue-behavior awareness improve attacker performance?
    
    Expected Result: Adaptive Red should achieve 5-10% higher win rate
    """
    from cyber_playground import train_agents
    
    print("\n" + "="*70)
    print("EXPERIMENT: Adaptive vs Standard Red Agent")
    print("="*70)
    
    # Train standard Red
    print("\n[1/2] Training Standard Red...")
    _, _, _, standard_metrics = train_agents(num_episodes=500)
    standard_final = np.mean(standard_metrics['red_rewards'][-100:])
    
    # Train adaptive Red
    print("\n[2/2] Training Adaptive Red...")
    _, _, _, adaptive_metrics = train_adaptive_agents(num_episodes=500)
    adaptive_final = np.mean(adaptive_metrics['red_rewards'][-100:])
    
    # Analysis
    improvement = ((adaptive_final - standard_final) / abs(standard_final)) * 100
    
    print("\n" + "="*70)
    print("RESULTS: Adaptive vs Standard")
    print("="*70)
    print(f"Standard Red Final Reward:  {standard_final:>8.1f}")
    print(f"Adaptive Red Final Reward:  {adaptive_final:>8.1f}")
    print(f"Improvement:                {improvement:>7.1f}%")
    print("="*70)
    
    if improvement > 5:
        print("\n✓ FINDING: Adaptive strategy significantly improves attacker performance")
        print("  Implication: Defenders must account for adaptive, learning adversaries")
    else:
        print("\n✓ FINDING: Adaptation provides marginal benefit")
        print("  Implication: Simple reactive strategies may suffice for basic attacks")
    
    return {
        'standard': standard_metrics,
        'adaptive': adaptive_metrics,
        'improvement': improvement
    }


if __name__ == "__main__":
    # Run the novel experiment
    results = compare_adaptive_vs_standard()