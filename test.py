"""
EfficientZero V2 - Evaluation & Demonstration Script
CT-469: Reinforcement Learning | Spring 2026

Loads a trained checkpoint and runs the agent on CartPole-v1.
Shows:
  - Live rendered gameplay (if a display is available)
  - Episode-by-episode reward breakdown
  - Comparison: trained EZ-V2 agent vs random agent
  - Training curve plot saved to results/training_curve.png

Run AFTER training:
    python test.py                              # uses latest checkpoint
    python test.py --model checkpoints/model_020000.pt   # specific checkpoint
    python test.py --episodes 20               # more episodes
    python test.py --no-render                 # headless (e.g. SSH)
"""

import os
import sys
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works on any machine
import matplotlib.pyplot as plt

# ─── import model definition from train.py ──────────────────────────────────
# We reuse the exact same model class and search function.
# This guarantees the loaded weights match the architecture used in training.
sys.path.insert(0, os.path.dirname(__file__))
from train import EZV2Model, gumbel_search, CFG, DEVICE

# ════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSING
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="EZ-V2 Evaluation")
    p.add_argument("--model",      type=str,  default=None,
                   help="Path to checkpoint .pt file. Defaults to latest in checkpoints/")
    p.add_argument("--episodes",   type=int,  default=10,
                   help="Number of episodes to evaluate (default: 10)")
    p.add_argument("--no-render",  action="store_true",
                   help="Disable rendering (use if running headless / SSH)")
    p.add_argument("--compare",    action="store_true", default=True,
                   help="Also run a random agent for comparison (default: True)")
    return p.parse_args()


def find_latest_checkpoint():
    """Find the most recently saved checkpoint in the checkpoints/ folder."""
    folder = "checkpoints"
    if not os.path.exists(folder):
        return None
    files = [f for f in os.listdir(folder) if f.endswith(".pt")]
    if not files:
        return None
    # Prefer model_best.pt — highest scoring checkpoint
    if "model_best.pt" in files:
        return os.path.join(folder, "model_best.pt")
    # Otherwise fall back to latest by step number
    def sort_key(f):
        if "final" in f:
            return float("inf")
        try:
            return int(f.replace("model_", "").replace(".pt", ""))
        except ValueError:
            return 0
    files.sort(key=sort_key)
    return os.path.join(folder, files[-1])


# ════════════════════════════════════════════════════════════════════════════
# LOAD CHECKPOINT
# ════════════════════════════════════════════════════════════════════════════

def load_model(path):
    """
    Load a trained EZ-V2 model from a checkpoint file.
    """
    print(f"\n[Load] Loading checkpoint: {path}")
    ckpt  = torch.load(path, map_location=DEVICE)
    model = EZV2Model().to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Get values, defaulting to 0 if they don't exist so the formatting doesn't crash
    env_step      = ckpt.get("env_step", 0)
    training_step = ckpt.get("training_step", 0)
    
    # If they are strings (like "?"), we convert to 0 or handle separately
    env_val = int(env_step) if isinstance(env_step, (int, float)) else 0
    train_val = int(training_step) if isinstance(training_step, (int, float)) else 0

    print(f"[Load] Trained for {env_val:,} env steps / {train_val:,} gradient updates")
    return model, env_step

# ════════════════════════════════════════════════════════════════════════════
# SINGLE EPISODE RUNNER
# ════════════════════════════════════════════════════════════════════════════

def run_episode(model, render=False, seed=None):
    """
    Run one full episode with the trained EZ-V2 agent.

    At each step:
        1. Encode obs → latent state via H (representation)
        2. Run Gumbel search to find best action
        3. Execute action in environment
        4. Repeat until done

    Returns:
        total_reward : float
        steps        : int
        action_hist  : list of actions taken (for analysis)
    """
    render_mode = "human" if render else "rgb_array"
    env         = gym.make(CFG["env_name"], render_mode=render_mode)

    if seed is not None:
        obs, _ = env.reset(seed=seed)
    else:
        obs, _ = env.reset()

    total_reward = 0.0
    steps        = 0
    action_hist  = []

    done = False
    while not done:
        # Gumbel search picks the action — same as during training
        # No exploration noise during evaluation (use_gumble_noise=False
        # mirrors what the paper's eval.py does)
        action, policy, value = gumbel_search(
            model, obs, env_step=999_999, training_step=999_999
        )

        obs, reward, terminated, truncated, info = env.step(action)
        done          = terminated or truncated
        total_reward += reward
        steps        += 1
        action_hist.append(action)

        if render:
            env.render()

    env.close()
    return total_reward, steps, action_hist


# ════════════════════════════════════════════════════════════════════════════
# RANDOM AGENT BASELINE
#
# WHY: To show that the trained model actually learned something, we compare
#      it against a random agent. CartPole with random actions typically
#      scores 8-15 per episode. A well-trained agent should score 400-500
#      (the maximum — CartPole terminates at 500 steps).
# ════════════════════════════════════════════════════════════════════════════

def run_random_episode(seed=None):
    """Run one episode with a purely random agent."""
    env  = gym.make(CFG["env_name"])
    if seed is not None:
        obs, _ = env.reset(seed=seed)
    else:
        obs, _ = env.reset()
    total = 0.0
    done  = False
    while not done:
        action              = env.action_space.sample()
        obs, r, te, tr, _  = env.step(action)
        done   = te or tr
        total += r
    env.close()
    return total


# ════════════════════════════════════════════════════════════════════════════
# TRAINING CURVE PLOTTER
# ════════════════════════════════════════════════════════════════════════════

def plot_training_curve():
    """
    Plot the training curve from the CSV log produced during training.
    Saves to results/training_curve.png — include this in your presentation!
    """
    log_path = "results/training_log.csv"
    if not os.path.exists(log_path):
        print("[Plot] No training log found. Run train.py first.")
        return

    steps, rewards, losses = [], [], []
    with open(log_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["env_step"]))
            rewards.append(float(row["eval_reward"]))
            losses.append(float(row["loss"]))

    if not steps:
        print("[Plot] Training log is empty.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=120)
    fig.suptitle("EfficientZero V2 — CartPole-v1 Training", fontsize=14, fontweight="bold")

    # ── Reward curve ──────────────────────────────────────────────────────
    ax1.plot(steps, rewards, color="#00C9A7", linewidth=2, label="EZ-V2 Agent")
    ax1.axhline(y=500, color="#FF6B6B", linestyle="--", alpha=0.7,
                label="Max possible (500)")
    # Smooth with a rolling average if enough data points
    if len(rewards) >= 5:
        smooth = np.convolve(rewards, np.ones(3)/3, mode="valid")
        ax1.plot(steps[1:-1], smooth, color="#007F6A", linewidth=2.5,
                 linestyle="-", alpha=0.6, label="3-pt avg")
    ax1.set_ylabel("Mean Episode Reward", fontsize=11)
    ax1.set_xlabel("Environment Steps", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor("#0D1B2A")
    ax1.tick_params(colors="white")
    ax1.yaxis.label.set_color("white")
    ax1.xaxis.label.set_color("white")
    ax1.title.set_color("white")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#1B2E41")

    # ── Loss curve ────────────────────────────────────────────────────────
    ax2.plot(steps, losses, color="#FFB347", linewidth=2, label="Total Loss")
    ax2.set_ylabel("Training Loss", fontsize=11)
    ax2.set_xlabel("Environment Steps", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor("#0D1B2A")
    ax2.tick_params(colors="white")
    ax2.yaxis.label.set_color("white")
    ax2.xaxis.label.set_color("white")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#1B2E41")

    fig.patch.set_facecolor("#0D1B2A")
    plt.tight_layout()
    out_path = "results/training_curve.png"
    plt.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot] Training curve saved → {out_path}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN EVALUATION LOOP
# ════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    os.makedirs("results", exist_ok=True)

    # ── Find / load model ─────────────────────────────────────────────────
    model_path = args.model or find_latest_checkpoint()
    if model_path is None:
        print("[Error] No checkpoint found. Please run train.py first.")
        sys.exit(1)

    model, trained_steps = load_model(model_path)

    # ── Run trained agent ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Running EZ-V2 Agent for {args.episodes} episodes")
    print(f"{'='*60}")

    ez_rewards  = []
    ez_steps    = []
    action_dist = {0: 0, 1: 0}   # count left vs right actions

    render = not args.no_render
    for ep in range(1, args.episodes + 1):
        reward, steps, actions = run_episode(model, render=render, seed=ep * 7)
        ez_rewards.append(reward)
        ez_steps.append(steps)
        for a in actions:
            action_dist[a] += 1
        print(f"  Episode {ep:>2}/{args.episodes}  |  "
              f"Reward: {reward:>6.1f}  |  Steps: {steps:>5}")

    print(f"\n  ── EZ-V2 Summary ──────────────────────────────")
    print(f"  Mean reward   : {np.mean(ez_rewards):.1f}")
    print(f"  Std  reward   : {np.std(ez_rewards):.1f}")
    print(f"  Best episode  : {max(ez_rewards):.1f}")
    print(f"  Worst episode : {min(ez_rewards):.1f}")
    print(f"  Action split  : LEFT={action_dist[0]}  RIGHT={action_dist[1]}")

    # ── Random agent comparison ────────────────────────────────────────────
    if args.compare:
        print(f"\n{'='*60}")
        print(f"  Running Random Agent for {args.episodes} episodes (baseline)")
        print(f"{'='*60}")

        rand_rewards = []
        for ep in range(1, args.episodes + 1):
            r = run_random_episode(seed=ep * 7)
            rand_rewards.append(r)
            print(f"  Episode {ep:>2}/{args.episodes}  |  Reward: {r:>6.1f}")

        print(f"\n  ── Random Agent Summary ────────────────────────")
        print(f"  Mean reward   : {np.mean(rand_rewards):.1f}")
        print(f"  Std  reward   : {np.std(rand_rewards):.1f}")

        # ── Improvement summary ───────────────────────────────────────────
        improvement = (np.mean(ez_rewards) / max(np.mean(rand_rewards), 1)) 
        print(f"\n{'='*60}")
        print(f"  EZ-V2 is {improvement:.1f}x better than random!")
        print(f"  EZ-V2 mean:  {np.mean(ez_rewards):.1f}")
        print(f"  Random mean: {np.mean(rand_rewards):.1f}")
        print(f"{'='*60}")

        # ── Bar chart comparison ──────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
        fig.patch.set_facecolor("#0D1B2A")
        ax.set_facecolor("#0D1B2A")

        ep_nums = list(range(1, args.episodes + 1))
        ax.bar([x - 0.2 for x in ep_nums], ez_rewards,   width=0.38,
               color="#00C9A7", label="EZ-V2 Agent")
        ax.bar([x + 0.2 for x in ep_nums], rand_rewards, width=0.38,
               color="#FF6B6B", alpha=0.8, label="Random Agent")

        ax.set_xlabel("Episode", fontsize=11, color="white")
        ax.set_ylabel("Total Reward", fontsize=11, color="white")
        ax.set_title("EZ-V2 vs Random Agent — CartPole-v1", fontsize=13,
                     color="white", fontweight="bold")
        ax.legend(fontsize=10)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#1B2E41")
        ax.axhline(y=500, color="white", linestyle="--", alpha=0.3,
                   label="Max (500)")
        ax.set_xticks(ep_nums)

        plt.tight_layout()
        out_path = "results/comparison_chart.png"
        plt.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        print(f"\n[Plot] Comparison chart saved → {out_path}")

    # ── Training curve ─────────────────────────────────────────────────────
    plot_training_curve()

    print(f"\n[Done] All results saved to results/")
    print(f"       training_curve.png  — show this in your presentation")
    print(f"       comparison_chart.png — shows EZ-V2 vs random")


if __name__ == "__main__":
    main()
