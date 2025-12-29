"""
Advanced Analysis Tools for CyberMARL
Provides deep insights into agent behavior and strategy evolution
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from collections import defaultdict


class StrategyAnalyzer:
    """
    Analyzes learned strategies and decision patterns
    """
    
    def __init__(self, red_qtable, blue_qtable):
        self.red_q = red_qtable
        self.blue_q = blue_qtable
        
    def analyze_action_preferences(self):
        """Analyze which actions agents prefer in different states"""
        red_prefs = {
            'EXPLOIT': 0,
            'MOVE': 0,
            'WAIT': 0
        }
        
        blue_prefs = {
            'MONITOR': 0,
            'QUARANTINE': 0,
            'PATCH': 0
        }
        
        # Count preferred actions across all states
        for state in range(len(self.red_q)):
            red_best = np.argmax(self.red_q[state])
            blue_best = np.argmax(self.blue_q[state])
            
            red_actions = ['EXPLOIT', 'MOVE', 'WAIT']
            blue_actions = ['MONITOR', 'QUARANTINE', 'PATCH']
            
            red_prefs[red_actions[red_best]] += 1
            blue_prefs[blue_actions[blue_best]] += 1
        
        return red_prefs, blue_prefs
    
    def plot_action_distribution(self, save_path='results/plots'):
        """Visualize action preference distribution"""
        red_prefs, blue_prefs = self.analyze_action_preferences()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Red Agent
        actions = list(red_prefs.keys())
        values = list(red_prefs.values())
        colors_red = ['#e74c3c', '#c0392b', '#922b21']
        
        ax1.bar(actions, values, color=colors_red, edgecolor='black', linewidth=2)
        ax1.set_title('Red Agent Action Preferences', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Times Selected as Best Action', fontsize=11)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add percentage labels
        total_red = sum(values)
        for i, v in enumerate(values):
            pct = (v / total_red) * 100
            ax1.text(i, v + 5, f'{pct:.1f}%', ha='center', fontweight='bold')
        
        # Blue Agent
        actions = list(blue_prefs.keys())
        values = list(blue_prefs.values())
        colors_blue = ['#3498db', '#2980b9', '#1f618d']
        
        ax2.bar(actions, values, color=colors_blue, edgecolor='black', linewidth=2)
        ax2.set_title('Blue Agent Action Preferences', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Times Selected as Best Action', fontsize=11)
        ax2.grid(axis='y', alpha=0.3)
        
        # Add percentage labels
        total_blue = sum(values)
        for i, v in enumerate(values):
            pct = (v / total_blue) * 100
            ax2.text(i, v + 5, f'{pct:.1f}%', ha='center', fontweight='bold')
        
        plt.tight_layout()
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f'{save_path}/action_distribution.png', dpi=300, bbox_inches='tight')
        print(f"✓ Action distribution saved to {save_path}/action_distribution.png")
        plt.close()
        
        return red_prefs, blue_prefs
    
    def analyze_q_value_statistics(self):
        """Get statistical summary of Q-values"""
        red_stats = {
            'mean': np.mean(self.red_q),
            'std': np.std(self.red_q),
            'min': np.min(self.red_q),
            'max': np.max(self.red_q),
            'median': np.median(self.red_q)
        }
        
        blue_stats = {
            'mean': np.mean(self.blue_q),
            'std': np.std(self.blue_q),
            'min': np.min(self.blue_q),
            'max': np.max(self.blue_q),
            'median': np.median(self.blue_q)
        }
        
        return red_stats, blue_stats
    
    def find_critical_states(self, top_n=5):
        """Identify states with highest Q-value variance (most strategic)"""
        red_variances = np.var(self.red_q, axis=1)
        blue_variances = np.var(self.blue_q, axis=1)
        
        red_critical = np.argsort(red_variances)[-top_n:][::-1]
        blue_critical = np.argsort(blue_variances)[-top_n:][::-1]
        
        return red_critical, blue_critical
    
    def generate_strategy_report(self, save_path='results/analysis_report.txt'):
        """Generate comprehensive strategy analysis report"""
        report = []
        report.append("="*70)
        report.append("CyberMARL STRATEGY ANALYSIS REPORT")
        report.append("="*70)
        report.append("")
        
        # Action preferences
        red_prefs, blue_prefs = self.analyze_action_preferences()
        
        report.append("1. ACTION PREFERENCES")
        report.append("-" * 70)
        report.append("Red Agent Strategy:")
        for action, count in red_prefs.items():
            pct = (count / sum(red_prefs.values())) * 100
            report.append(f"  {action:15s}: {count:4d} states ({pct:5.1f}%)")
        
        report.append("")
        report.append("Blue Agent Strategy:")
        for action, count in blue_prefs.items():
            pct = (count / sum(blue_prefs.values())) * 100
            report.append(f"  {action:15s}: {count:4d} states ({pct:5.1f}%)")
        
        report.append("")
        
        # Q-value statistics
        red_stats, blue_stats = self.analyze_q_value_statistics()
        
        report.append("2. Q-VALUE STATISTICS")
        report.append("-" * 70)
        report.append(f"Red Agent:")
        report.append(f"  Mean Q-value:   {red_stats['mean']:8.2f}")
        report.append(f"  Std Deviation:  {red_stats['std']:8.2f}")
        report.append(f"  Min Q-value:    {red_stats['min']:8.2f}")
        report.append(f"  Max Q-value:    {red_stats['max']:8.2f}")
        report.append(f"  Median:         {red_stats['median']:8.2f}")
        
        report.append("")
        report.append(f"Blue Agent:")
        report.append(f"  Mean Q-value:   {blue_stats['mean']:8.2f}")
        report.append(f"  Std Deviation:  {blue_stats['std']:8.2f}")
        report.append(f"  Min Q-value:    {blue_stats['min']:8.2f}")
        report.append(f"  Max Q-value:    {blue_stats['max']:8.2f}")
        report.append(f"  Median:         {blue_stats['median']:8.2f}")
        
        report.append("")
        
        # Critical states
        red_critical, blue_critical = self.find_critical_states()
        
        report.append("3. CRITICAL DECISION STATES")
        report.append("-" * 70)
        report.append("States with highest strategic importance (variance):")
        report.append(f"Red Agent critical states: {red_critical.tolist()}")
        report.append(f"Blue Agent critical states: {blue_critical.tolist()}")
        
        report.append("")
        report.append("="*70)
        
        # Save report
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        with open(save_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"✓ Strategy report saved to {save_path}")
        
        # Also print to console
        print('\n'.join(report))
        
        return '\n'.join(report)


class ComparisonAnalyzer:
    """
    Compare results from different training runs
    """
    
    def __init__(self):
        self.runs = []
    
    def add_run(self, name, metrics):
        """Add a training run for comparison"""
        self.runs.append({
            'name': name,
            'metrics': metrics
        })
    
    def plot_comparison(self, save_path='results/plots/comparison.png'):
        """Plot comparison of multiple runs"""
        if len(self.runs) < 2:
            print("Need at least 2 runs to compare")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Run Comparison', fontsize=16, fontweight='bold')
        
        for run in self.runs:
            name = run['name']
            metrics = run['metrics']
            
            # Red rewards
            window = 50
            red_ma = np.convolve(metrics['red_rewards'], np.ones(window)/window, mode='valid')
            axes[0, 0].plot(range(window, len(metrics['red_rewards']) + 1), red_ma, 
                          label=name, linewidth=2, alpha=0.8)
            
            # Blue rewards
            blue_ma = np.convolve(metrics['blue_rewards'], np.ones(window)/window, mode='valid')
            axes[0, 1].plot(range(window, len(metrics['blue_rewards']) + 1), blue_ma,
                          label=name, linewidth=2, alpha=0.8)
            
            # Final distribution
            axes[1, 0].hist(metrics['red_rewards'][-100:], bins=20, alpha=0.5, label=name)
            axes[1, 1].hist(metrics['blue_rewards'][-100:], bins=20, alpha=0.5, label=name)
        
        axes[0, 0].set_title('Red Agent Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward (MA-50)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('Blue Agent Rewards')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Reward (MA-50)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_title('Red Final 100 Episodes Distribution')
        axes[1, 0].set_xlabel('Reward')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        axes[1, 1].set_title('Blue Final 100 Episodes Distribution')
        axes[1, 1].set_xlabel('Reward')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison plot saved to {save_path}")
        plt.close()


def analyze_trained_models():
    """Main analysis function"""
    print("\n" + "="*70)
    print("RUNNING ADVANCED ANALYSIS")
    print("="*70)
    
    try:
        # Load Q-tables
        red_q = np.load('results/trained_models/red_agent_qtable.npy')
        blue_q = np.load('results/trained_models/blue_agent_qtable.npy')
        
        # Create analyzer
        analyzer = StrategyAnalyzer(red_q, blue_q)
        
        # Generate visualizations
        print("\n1. Analyzing action preferences...")
        analyzer.plot_action_distribution()
        
        # Generate report
        print("\n2. Generating strategy report...")
        analyzer.generate_strategy_report()
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print("="*70)
        
    except FileNotFoundError:
        print("Error: Trained models not found. Train agents first with:")
        print("  python cyber_playground.py --train --episodes 1000")


if __name__ == "__main__":
    analyze_trained_models()