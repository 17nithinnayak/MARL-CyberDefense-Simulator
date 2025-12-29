"""
Experiment Runner for CyberMARL
Runs multiple configurations and compares results
"""

import subprocess
import json
import os
import numpy as np
from advanced_analysis import ComparisonAnalyzer


def run_experiment(name, episodes, alpha=0.1, gamma=0.9):
    """Run a single training experiment"""
    print(f"\n{'='*70}")
    print(f"Running Experiment: {name}")
    print(f"{'='*70}")
    print(f"Episodes: {episodes}")
    print(f"Alpha (α): {alpha}")
    print(f"Gamma (γ): {gamma}")
    print()
    
    # Run training
    cmd = f"python cyber_playground.py --train --episodes {episodes}"
    subprocess.run(cmd, shell=True)
    
    # Load results
    try:
        with open('results/logs/training_metrics.json', 'r') as f:
            metrics = json.load(f)
        
        # Backup results
        backup_dir = f'results/experiments/{name}'
        os.makedirs(backup_dir, exist_ok=True)
        
        # Save metrics
        with open(f'{backup_dir}/metrics.json', 'w') as f:
            json.dump(metrics, f)
        
        # Copy Q-tables
        os.system(f'cp results/trained_models/red_agent_qtable.npy {backup_dir}/')
        os.system(f'cp results/trained_models/blue_agent_qtable.npy {backup_dir}/')
        
        print(f"✓ Results saved to {backup_dir}")
        
        return metrics
        
    except Exception as e:
        print(f"Error loading results: {e}")
        return None


def compare_experiments():
    """Compare all saved experiments"""
    print("\n" + "="*70)
    print("COMPARING EXPERIMENTS")
    print("="*70)
    
    analyzer = ComparisonAnalyzer()
    
    exp_dir = 'results/experiments'
    if not os.path.exists(exp_dir):
        print("No experiments found. Run experiments first.")
        return
    
    # Load all experiments
    for exp_name in os.listdir(exp_dir):
        metrics_file = f'{exp_dir}/{exp_name}/metrics.json'
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            analyzer.add_run(exp_name, metrics)
            print(f"✓ Loaded: {exp_name}")
    
    # Generate comparison
    if len(analyzer.runs) >= 2:
        analyzer.plot_comparison()
        print("\n✓ Comparison visualization saved!")
    else:
        print("\nNeed at least 2 experiments to compare.")


def run_ablation_study():
    """
    Run ablation study: Test impact of zero-day exploits
    """
    print("\n" + "="*70)
    print("ABLATION STUDY: Impact of Zero-Day Exploits")
    print("="*70)
    
    # Baseline (no zero-day)
    print("\nBaseline: Zero-Day Probability = 0%")
    # Note: You'd need to modify ZERO_DAY_PROB in cyber_playground.py
    
    # With zero-day
    print("\nEnhanced: Zero-Day Probability = 25%")
    # Current version has this
    
    print("\nRun both versions manually and compare:")
    print("1. Set ZERO_DAY_PROB = 0.0 in cyber_playground.py")
    print("2. Run: python run_experiments.py baseline")
    print("3. Set ZERO_DAY_PROB = 0.25 in cyber_playground.py")
    print("4. Run: python run_experiments.py zero_day")
    print("5. Run: python run_experiments.py --compare")


def main():
    """Main experiment runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description='CyberMARL Experiment Runner')
    parser.add_argument('experiment', nargs='?', default='default',
                       help='Experiment name (default, baseline, zero_day)')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--compare', action='store_true',
                       help='Compare all experiments')
    parser.add_argument('--ablation', action='store_true',
                       help='Run ablation study guide')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_experiments()
    elif args.ablation:
        run_ablation_study()
    else:
        run_experiment(args.experiment, args.episodes)
        
        # Auto-run analysis
        print("\nRunning post-training analysis...")
        from advanced_analysis import analyze_trained_models
        analyze_trained_models()


if __name__ == "__main__":
    main()