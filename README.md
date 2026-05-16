EfficientZero V2 - Simplified Reproduction (CartPole-v1)

A lightweight educational implementation of **EfficientZero V2** (ICML 2024) focused on the core algorithmic ideas: **world model learning**, **Sampling-Based Gumbel Search**, **Sequential Halving**, and **Search-Based Value Estimation (SVE)** with mixed value targets.


Overview

This repository contains our reproduction of EfficientZero V2 for the **Gymnasium CartPole-v1** environment. While the original paper demonstrates state-of-the-art performance across 66 diverse tasks (Atari, DMControl proprioceptive & vision), our version is intentionally simplified for clarity, education, and single-machine execution.

**Key Implemented Ideas**:
- Latent world model (Representation + Dynamics + Policy + Value)
- Sampling-Based Gumbel Search with Sequential Halving
- Mixed value targets (TD for fresh/early data + SVE for stale replay data)
- Replay buffer with n-step TD targets
- Policy improvement via search


Features

- **Train.py**: Full training pipeline with world model, Gumbel search, and mixed targets
- **Test.py**: Evaluation script with live rendering, random agent baseline, and comparison charts
- Automatic checkpointing (`model_best.pt`)
- Training logs + visualization (training curve + comparison bar chart)
- Clean, well-documented code with explanations tied to the paper



Results

Our simplified agent successfully learns to balance the pole:

- **Best mean evaluation reward**: ~294 (at 34k steps)
- **Final evaluation**: Significantly outperforms random agent (~5x better)
- Achieves multiple **500-step** (maximum) episodes
- Training curve shows clear learning with some instability (typical for small-scale RL)

**Visualizations** are automatically saved in the `results/` folder.



## 🛠️ Installation

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd EZV2-CartPole

# 2. Create environment (recommended)
conda create -n ezv2 python=3.10
conda activate ezv2

# 3. Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # or cpu version
pip install gymnasium matplotlib numpy
```



Usage

Training

```bash
python train.py
```

Training runs for 50,000 environment steps. Checkpoints and logs are saved automatically.


Evaluation

```bash
# Using best model
python test.py

# Specific checkpoint
python test.py --model checkpoints/model_best.pt

# More episodes + no rendering (headless)
python test.py --episodes 20 --no-render
```


Project Structure

```
├── train.py                    
├── test.py                     
├── checkpoints/            
├── results/
│   ├── training_log.csv
│   ├── training_curve.png
│   └── comparison_chart.png
```


Key Differences from Original Paper

| Aspect                | Original EZ-V2                  | Our Implementation              |
|-----------------------|---------------------------------|---------------------------------|
| Environments          | 66 tasks (Atari, DMControl, etc.) | Only CartPole-v1               |
| Action Space          | Discrete + Continuous           | Discrete (2 actions)           |
| Architecture          | Large-scale distributed         | Single script, compact MLP     |
| Search                | Full Sampling Gumbel + SVE      | Simplified Gumbel-Top-k + Halving |
| Scale                 | Millions of steps, multi-GPU    | 50k steps, single machine      |


Implementation Highlights

- **Gumbel Search**: Uses Gumbel-Top-k + Sequential Halving for efficient action selection
- **Mixed Value Target**: TD for early/fresh data, Search-Based Value Estimation for stale data
- **World Model**: Representation (`H`), Dynamics (`G`), Policy (`P`), Value (`V`)
- **Losses**: Reward + Policy + Value + Consistency (SimSiam-style)

