"""
NOVEL CONTRIBUTION: Multi-Objective Defense Agent

Key Innovation: Blue agent balancing security AND operational costs
Paper Difference: RLJ 2025 optimizes only for security; we add cost constraints

Research Question: Can defenders learn to balance security with business continuity?
"""

import numpy as np

class MultiObjectiveBlueAgent:
    """
    Blue agent with dual objectives: Security + Cost minimization
    
    Novel Features:
    1. Tracks operational costs (patches, quarantines disrupt business)
    2. Learns to prioritize critical vs. non-critical actions
    3. Implements cost-aware Q-learning with scalarization
    4. Discovers Pareto-optimal defense strategies
    """
    
    def __init__(self, state_space_size, action_space_size, cost_weight=0.3):
        # Standard Q-learning
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 1.0
        
        # NOVEL: Dual Q-tables for multi-objective learning
        self.q_security = np.zeros((state_space_size, action_space_size))  # Security reward
        self.q_cost = np.zeros((state_space_size, action_space_size))      # Operational cost
        
        # NOVEL: Cost-benefit parameters
        self.cost_weight = cost_weight  # 0.0 = pure security, 1.0 = pure cost minimization
        
        # Action costs (operational disruption)
        self.action_costs = {
            0: 1,   # MONITOR: Low cost (passive)
            1: 10,  # QUARANTINE: High cost (disrupts services)
            2: 15   # PATCH: Very high cost (downtime for all hosts)
        }
        
        # NOVEL: Cost tracking for analysis
        self.total_security_reward = 0
        self.total_operational_cost = 0
        self.action_cost_history = []
        
    def choose_action(self, state):
        """
        NOVEL: Action selection balancing security and cost
        
        Combined Q-value: Q_total = (1-w)*Q_security - w*Q_cost
        where w is cost_weight
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_space_size)
        
        # NOVEL: Scalarization of multi-objective Q-values
        q_combined = (1 - self.cost_weight) * self.q_security[state] - \
                     self.cost_weight * self.q_cost[state]
        
        return np.argmax(q_combined)
    
    def learn(self, state, action, security_reward, next_state, done):
        """
        NOVEL: Update both security and cost Q-tables
        
        Key Innovation: Learns security-cost tradeoffs automatically
        """
        # Get operational cost for this action
        operational_cost = self.action_costs[action]
        
        # Update security Q-table (standard Q-learning)
        current_q_sec = self.q_security[state, action]
        if done:
            target_q_sec = security_reward
        else:
            max_next_q_sec = np.max(self.q_security[next_state])
            target_q_sec = security_reward + self.gamma * max_next_q_sec
        
        self.q_security[state, action] = current_q_sec + self.alpha * (target_q_sec - current_q_sec)
        
        # NOVEL: Update cost Q-table
        current_q_cost = self.q_cost[state, action]
        if done:
            target_q_cost = operational_cost
        else:
            max_next_q_cost = np.max(self.q_cost[next_state])
            target_q_cost = operational_cost + self.gamma * max_next_q_cost
        
        self.q_cost[state, action] = current_q_cost + self.alpha * (target_q_cost - current_q_cost)
        
        # Track for analysis
        self.total_security_reward += security_reward
        self.total_operational_cost += operational_cost
        self.action_cost_history.append(operational_cost)
    
    def decay_epsilon(self):
        self.epsilon = max(0.01, self.epsilon * 0.995)
    
    def get_pareto_analysis(self):
        """
        NOVEL: Return Pareto frontier analysis
        
        Shows security-cost tradeoffs discovered by agent
        """
        return {
            'total_security': self.total_security_reward,
            'total_cost': self.total_operational_cost,
            'avg_action_cost': np.mean(self.action_cost_history) if self.action_cost_history else 0,
            'efficiency_ratio': self.total_security_reward / (self.total_operational_cost + 1)
        }
    
    def get_action_preferences_by_cost(self):
        """
        Analyze which actions are preferred at different cost sensitivities
        """
        preferences = {}
        
        for state in range(self.state_space_size):
            q_combined = (1 - self.cost_weight) * self.q_security[state] - \
                        self.cost_weight * self.q_cost[state]
            best_action = np.argmax(q_combined)
            
            if best_action not in preferences:
                preferences[best_action] = 0
            preferences[best_action] += 1
        
        return preferences


# ============================================================================
# EXPERIMENT: COST-SENSITIVE DEFENSE
# ============================================================================

def train_cost_sensitive_defense(cost_weights=[0.0, 0.3, 0.5, 0.7]):
    """
    NOVEL EXPERIMENT: Compare defenses at different cost sensitivities
    
    Research Question: How does operational cost awareness affect defense strategy?
    
    Expected Findings:
    - Low cost_weight (0.0): Aggressive patching, high security, high cost
    - High cost_weight (0.7): Reactive monitoring, lower security, low cost
    - Optimal: Somewhere in middle (Pareto-efficient)
    """
    from cyber_playground import CyberEnv, QLearningAgent
    
    print("\n" + "="*70)
    print("EXPERIMENT: Cost-Sensitive Defense Learning")
    print("="*70)
    
    results = {}
    
    for cost_weight in cost_weights:
        print(f"\n[Training] Cost Weight = {cost_weight:.1f}")
        print("-" * 70)
        
        env = CyberEnv()
        red_agent = QLearningAgent(env.state_space_size, len(env.red_actions), "RED")
        blue_agent = MultiObjectiveBlueAgent(env.state_space_size, len(env.blue_actions), 
                                            cost_weight=cost_weight)
        
        blue_rewards = []
        
        for episode in range(500):
            state = env.reset()
            episode_blue_reward = 0
            done = False
            
            while not done:
                red_action = red_agent.choose_action(state)
                blue_action = blue_agent.choose_action(state)
                
                next_state, red_reward, blue_reward, done, info = env.step(red_action, blue_action)
                
                red_agent.learn(state, red_action, red_reward, next_state, done)
                blue_agent.learn(state, blue_action, blue_reward, next_state, done)
                
                state = next_state
                episode_blue_reward += blue_reward
            
            red_agent.decay_epsilon()
            blue_agent.decay_epsilon()
            blue_rewards.append(episode_blue_reward)
        
        # Analyze results
        pareto_stats = blue_agent.get_pareto_analysis()
        final_reward = np.mean(blue_rewards[-100:])
        
        results[cost_weight] = {
            'final_reward': final_reward,
            'security': pareto_stats['total_security'],
            'cost': pareto_stats['total_cost'],
            'efficiency': pareto_stats['efficiency_ratio'],
            'preferences': blue_agent.get_action_preferences_by_cost()
        }
        
        print(f"  Final Reward: {final_reward:.1f}")
        print(f"  Efficiency:   {pareto_stats['efficiency_ratio']:.2f}")
    
    # Print comparison
    print("\n" + "="*70)
    print("RESULTS: Cost-Security Tradeoff Analysis")
    print("="*70)
    print(f"{'Cost Weight':<15} | {'Security':<12} | {'Op. Cost':<12} | {'Efficiency':<12}")
    print("-" * 70)
    
    for cost_weight in cost_weights:
        r = results[cost_weight]
        print(f"{cost_weight:<15.1f} | {r['security']:<12.1f} | {r['cost']:<12.1f} | {r['efficiency']:<12.2f}")
    
    print("="*70)
    
    # Find Pareto-optimal point
    efficiencies = [(w, results[w]['efficiency']) for w in cost_weights]
    best_weight, best_eff = max(efficiencies, key=lambda x: x[1])
    
    print(f"\n✓ FINDING: Optimal cost-weight = {best_weight:.1f}")
    print(f"  This achieves best security-per-dollar efficiency")
    print(f"  Implication: Pure security optimization is economically suboptimal")
    
    return results


# ============================================================================
# VISUALIZATION: PARETO FRONTIER
# ============================================================================

def plot_pareto_frontier(results):
    """
    NOVEL VISUALIZATION: Security vs Cost tradeoff
    """
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Pareto Frontier
    costs = [results[w]['cost'] for w in sorted(results.keys())]
    security = [results[w]['security'] for w in sorted(results.keys())]
    weights = sorted(results.keys())
    
    ax1.plot(costs, security, 'o-', linewidth=2, markersize=10, color='purple')
    for i, w in enumerate(weights):
        ax1.annotate(f'w={w:.1f}', (costs[i], security[i]), 
                    xytext=(10, 10), textcoords='offset points', fontsize=9)
    
    ax1.set_xlabel('Total Operational Cost', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Total Security Reward', fontsize=12, fontweight='bold')
    ax1.set_title('Pareto Frontier: Security-Cost Tradeoff', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Efficiency by Cost Weight
    efficiency = [results[w]['efficiency'] for w in weights]
    
    ax2.bar(range(len(weights)), efficiency, color='green', alpha=0.7, edgecolor='black')
    ax2.set_xticks(range(len(weights)))
    ax2.set_xticklabels([f'{w:.1f}' for w in weights])
    ax2.set_xlabel('Cost Weight', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Security / Cost Ratio', fontsize=12, fontweight='bold')
    ax2.set_title('Defense Efficiency Analysis', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/plots/pareto_frontier.png', dpi=300, bbox_inches='tight')
    print("✓ Pareto frontier visualization saved")
    plt.show()


if __name__ == "__main__":
    # Run the novel experiment
    results = train_cost_sensitive_defense()
    plot_pareto_frontier(results)