# CyberMARL

**Multi-Agent Reinforcement Learning for Cyber Defense**

> Adaptive Red-Blue Team AI Simulation using Independent Q-Learning

## 🎯 Project Overview

CyberMARL is an adversarial multi-agent reinforcement learning environment where autonomous Red (attacker) and Blue (defender) agents learn optimal strategies through competitive self-play in a simulated network environment.

### Key Features
- ✅ Custom OpenAI Gym-style environment
- ✅ Independent Q-Learning (IQL) agents
- ✅ Zero-sum competitive dynamics
- ✅ Comprehensive visualization suite
- ✅ Research-backed implementation

## 🏗️ Architecture
```
Network: 3 hosts with sequential connectivity
Red Agent: Exploit, Move, Wait
Blue Agent: Monitor, Quarantine, Patch
State Space: 24 discrete states
Action Space: 3 actions per agent
```

## 📦 Installation
```bash
# Clone repository
git clone https://github.com/yourusername/CyberMARL.git
cd CyberMARL

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage
```bash
# Train agents
python cyber_playground.py --train --episodes 1000

# Run demo
python cyber_playground.py --demo

# Generate visualizations
python cyber_playground.py --visualize
```

## 📊 Results

Training results and visualizations will be saved in `results/` directory.

## 📚 Research Foundation

Based on: **Hierarchical Multi-agent Reinforcement Learning for Cyber Network Defense**  
*Reinforcement Learning Journal (RLJ), May 2025*

## 📄 License

MIT License - Academic Use

## 👨‍💻 Author

[Your Name] - CN Security Course Project

---

*Last Updated: December 2025*
>>>>>>> 161ea42 (Initial Commit)
