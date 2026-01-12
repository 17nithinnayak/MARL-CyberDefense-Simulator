"""
Generate Epsilon Decay Visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import json

# Option 1: Load from training metrics
try:
    with open('results/logs/training_metrics.json', 'r') as f:
        metrics = json.load(f)
    
    epsilon_history = metrics['epsilon_history']
    episodes = range(1, len(epsilon_history) + 1)
    
    print(f"✓ Loaded {len(epsilon_history)} epsilon values from training")
    
except FileNotFoundError:
    print("⚠ Training metrics not found. Generating theoretical epsilon decay...")
    
    # Option 2: Generate theoretical epsilon decay
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.995
    num_episodes = 1000
    
    epsilon_history = []
    epsilon = EPSILON_START
    
    for episode in range(num_episodes):
        epsilon_history.append(epsilon)
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    
    episodes = range(1, num_episodes + 1)

# Create the visualization
plt.figure(figsize=(10, 6))

# Plot epsilon decay
plt.plot(episodes, epsilon_history, 'g-', linewidth=2.5, label='Epsilon (ε)')

# Add threshold lines
plt.axhline(y=0.5, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='50% Exploration')
plt.axhline(y=0.1, color='red', linestyle='--', linewidth=1, alpha=0.7, label='10% Exploration')

# Annotations for key milestones
if len(epsilon_history) >= 100:
    plt.annotate(f'Episode 100\nε={epsilon_history[99]:.3f}',
                xy=(100, epsilon_history[99]),
                xytext=(150, epsilon_history[99] + 0.15),
                fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                arrowprops=dict(arrowstyle='->', lw=1.5))

if len(epsilon_history) >= 500:
    plt.annotate(f'Episode 500\nε={epsilon_history[499]:.3f}',
                xy=(500, epsilon_history[499]),
                xytext=(550, epsilon_history[499] + 0.1),
                fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                arrowprops=dict(arrowstyle='->', lw=1.5))

# Styling
plt.xlabel('Episode', fontsize=14, fontweight='bold')
plt.ylabel('Epsilon (Exploration Rate)', fontsize=14, fontweight='bold')
plt.title('Exploration Rate Decay Over Training', fontsize=16, fontweight='bold', pad=20)
plt.legend(fontsize=12, loc='upper right')
plt.grid(True, alpha=0.3, linestyle='--')
plt.ylim([0, 1.05])
plt.xlim([0, len(episodes)])

# Add shaded regions for exploration phases
plt.axhspan(0.5, 1.0, alpha=0.1, color='red', label='High Exploration')
plt.axhspan(0.1, 0.5, alpha=0.1, color='yellow')
plt.axhspan(0, 0.1, alpha=0.1, color='green', label='High Exploitation')

# Add text annotations for phases
plt.text(len(episodes) * 0.15, 0.75, 'EXPLORATION\nPhase', 
         fontsize=12, fontweight='bold', ha='center',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.text(len(episodes) * 0.5, 0.3, 'TRANSITION\nPhase', 
         fontsize=12, fontweight='bold', ha='center',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.text(len(episodes) * 0.85, 0.05, 'EXPLOITATION\nPhase', 
         fontsize=12, fontweight='bold', ha='center',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()

# Save the figure
plt.savefig('results/plots/epsilon_decay.png', dpi=300, bbox_inches='tight')
print("✓ Epsilon decay graph saved to results/plots/epsilon_decay.png")

plt.show()

# Print key statistics
print("\n" + "="*60)
print("EPSILON DECAY STATISTICS")
print("="*60)
print(f"Starting Epsilon: {epsilon_history[0]:.4f}")
print(f"Final Epsilon:    {epsilon_history[-1]:.4f}")
print(f"Total Decay:      {(epsilon_history[0] - epsilon_history[-1]):.4f}")
print(f"Decay Rate:       {((1 - epsilon_history[-1]/epsilon_history[0]) * 100):.1f}%")

# Find episode where epsilon drops below certain thresholds
for threshold in [0.5, 0.1, 0.05]:
    for i, eps in enumerate(epsilon_history):
        if eps < threshold:
            print(f"Episode when ε < {threshold}: {i+1}")
            break

print("="*60)