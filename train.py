"""
EfficientZero V2 - Simplified Training Script
CT-469: Reinforcement Learning | Spring 2026

Uses Gymnasium + CartPole-v1 (discrete) environment.
Implements the core EZ-V2 ideas:
  - World model (representation + dynamics + value/policy functions)
  - Gumbel-Top-k action sampling
  - Sequential Halving tree search
  - Mixed value target (TD for fresh data, search-based for stale data)
  - Replay buffer with priority sampling

Run:
    python train.py

Saves:
    checkpoints/model_best.pt    best model seen during training
    checkpoints/model_XXXXX.pt   every 5000 steps
    results/training_log.csv     reward + loss at every eval
"""

import os
import csv
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

# ─── reproducibility ────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ─── directories ────────────────────────────────────────────────────────────
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results",      exist_ok=True)

# ════════════════════════════════════════════════════════════════════════════
# 1.  HYPERPARAMETERS
# ════════════════════════════════════════════════════════════════════════════
CFG = {
    # environment
    "env_name":          "CartPole-v1",
    "obs_dim":           4,      # [x, x_dot, theta, theta_dot]
    "action_dim":        2,      # push left or push right

    # world model sizes
    # WHY 256: 128 was too small — the dynamics model couldn't distinguish
    # "about to fall" states from "stable" states with similar pole angles.
    "hidden_dim":        256,
    "mlp_layers":        3,

    # tree search
    "num_simulations":   20,     # WHY 20: with 2 actions, this gives ~10 visits
                                 # per action — enough for stable Q estimates
    "num_top_actions":   2,      # CartPole only has 2 actions, sample both
    "discount":          0.99,   # WHY 0.99 not 0.997: 0.997 under-values rewards
                                 # in short CartPole episodes (typically <500 steps)

    # mixed value target (Section 4.3 of paper)
    "td_steps":          10,     # WHY 10 not 5: 5-step TD only sees ~1 second of
                                 # gameplay. 10-step captures enough future to
                                 # learn the balance-reward relationship.
    "T1":                500,    # training steps before SVE activates
    "T2":                1000,   # transitions older than this use SVE not TD

    # replay buffer
    # WHY 20k not 50k: a 50k buffer fills with bad early-episode data that
    # drowns out good later data. 20k turns over faster, keeping training fresh.
    "buffer_size":       20_000,
    "batch_size":        256,    # larger batches = more stable gradient estimates
    "min_buffer_size":   200,    # WHY 200 not 500: start training sooner so
                                 # model begins guiding decisions earlier

    # optimiser
    # WHY 1e-3 not 3e-4: 3e-4 converges too slowly for a 50k step budget.
    # The LR scheduler then decays it as training stabilises.
    "lr":                1e-3,
    "weight_decay":      1e-4,
    "grad_clip":         10.0,

    # training
    "total_steps":       50_000,
    "unroll_steps":      3,      # WHY 3 not 5: 5-step unrolling made prediction
                                 # errors compound badly in early training. 3 is stable.

    # evaluation
    "eval_interval":     2000,
    "eval_episodes":     5,
    "save_interval":     5000,

    # loss coefficients
    "coeff_reward":      1.0,
    "coeff_policy":      1.0,
    "coeff_value":       1.0,    # WHY 1.0 not 0.25: 0.25 made value learning too slow
                                 # so Q-estimates in the search were consistently poor
    "coeff_consistency": 1.0,    # reduced from 2.0 to avoid drowning out policy loss

    # epsilon-greedy exploration
    # WHY ADD EPSILON-GREEDY ON TOP OF GUMBEL SEARCH:
    # Early in training, policy logits are near-random noise. Running Gumbel
    # search on random logits just wastes time — the dynamics model is also
    # untrained so the search can't improve on random anyway. Epsilon-greedy
    # skips the expensive search during early exploration and collects
    # diverse experience cheaply. Once epsilon drops, the search takes over.
    "epsilon_start":     1.0,
    "epsilon_end":       0.05,
    "epsilon_decay":     15_000, # linearly decay over first 15k steps
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Config] Device: {DEVICE}")
print(f"[Config] Environment: {CFG['env_name']}")
print(f"[Config] Total steps: {CFG['total_steps']:,}")


# ════════════════════════════════════════════════════════════════════════════
# 2.  WORLD MODEL
#     Four networks — H (representation), G (dynamics), P (policy), V (value)
# ════════════════════════════════════════════════════════════════════════════

def build_mlp(in_dim, out_dim, hidden_dim, num_layers, activation=nn.ReLU):
    """
    Build a Multi-Layer Perceptron.
    LayerNorm after each hidden layer stabilises training — especially
    important for the dynamics network which can see very different
    input scales across timesteps.
    """
    layers = [nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), activation()]
    for _ in range(num_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), activation()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class EZV2Model(nn.Module):
    """
    The complete EfficientZero V2 world model.

    WHY A WORLD MODEL:
        A world model lets the agent *imagine* future states without playing
        the real game. The Gumbel tree search uses G (dynamics) to simulate
        possible futures. This means one real game step can generate 20
        simulated steps of learning signal — that's the sample efficiency.
    """
    def __init__(self):
        super().__init__()
        obs   = CFG["obs_dim"]
        act   = CFG["action_dim"]
        hid   = CFG["hidden_dim"]
        depth = CFG["mlp_layers"]

        # H — Representation: raw observation → latent state
        self.representation  = build_mlp(obs, hid, hid, depth)

        # G — Dynamics: (latent state, action) → (next latent state, reward)
        self.dynamics_state  = build_mlp(hid + act, hid, hid, depth)
        self.dynamics_reward = build_mlp(hid, 1, hid, 1)

        # P — Policy: latent state → action logits
        self.policy = build_mlp(hid, act, hid, depth)

        # V — Value: latent state → expected future reward
        self.value  = build_mlp(hid, 1, hid, depth)

        # Projection heads for temporal consistency loss (from SimSiam)
        self.projector   = build_mlp(hid, hid, hid, 1)
        self.predictor   = build_mlp(hid, hid, hid, 1)
        self.target_proj = build_mlp(hid, hid, hid, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Zero-init output layers so predictions start near zero
        for net in [self.dynamics_reward, self.policy, self.value]:
            last = list(net.children())[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def represent(self, obs):
        return self.representation(obs)

    def dynamics(self, state, action_onehot):
        x          = torch.cat([state, action_onehot], dim=-1)
        next_state = self.dynamics_state(x)
        reward     = self.dynamics_reward(next_state).squeeze(-1)
        return next_state, reward

    def predict(self, state):
        return self.policy(state), self.value(state).squeeze(-1)

    def project(self, state, use_predictor=False):
        z = self.projector(state)
        return self.predictor(z) if use_predictor else z

    def target_project(self, state):
        with torch.no_grad():
            return self.target_proj(state)


# ════════════════════════════════════════════════════════════════════════════
# 3.  EXPLORATION SCHEDULE
# ════════════════════════════════════════════════════════════════════════════

def get_epsilon(env_step):
    """Linear decay: 1.0 → 0.05 over epsilon_decay steps."""
    progress = min(env_step / CFG["epsilon_decay"], 1.0)
    return CFG["epsilon_start"] + progress * (CFG["epsilon_end"] - CFG["epsilon_start"])


# ════════════════════════════════════════════════════════════════════════════
# 4.  GUMBEL SEARCH — core EZ-V2 algorithm
# ════════════════════════════════════════════════════════════════════════════

def gumbel_top_k(logits, k):
    """
    Sample K actions WITHOUT replacement using Gumbel-Top-k trick.

    Why Gumbel noise:
        argmax(g + logits) where g ~ Gumbel(0,1) is equivalent to
        sampling from the categorical distribution defined by the logits.
        Top-k ensures no action is sampled twice — no wasted simulations.
    """
    g      = -np.log(-np.log(np.random.uniform(1e-10, 1.0, len(logits))) + 1e-10)
    scores = g + logits
    top_k  = np.argsort(scores)[::-1][:k]
    return top_k.tolist(), g


def gumbel_search(model, obs, env_step, training_step, use_noise=True):
    """
    Sampling-Based Gumbel Search + Sequential Halving.

    1. Encode obs → latent state (H)
    2. Get policy logits + value (P, V)
    3. Sample K actions with Gumbel-Top-k
    4. Sequential Halving: visit candidates, eliminate worst half each round
    5. Return best action + improved policy distribution
    """
    model.eval()
    with torch.no_grad():
        obs_t         = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        state         = model.represent(obs_t)
        logits, value = model.predict(state)
        logits_np     = logits.squeeze(0).cpu().numpy()
        value_np      = value.item()

        K    = min(CFG["num_top_actions"], CFG["action_dim"])
        disc = CFG["discount"]

        if use_noise:
            top_k_actions, g_noise = gumbel_top_k(logits_np, K)
        else:
            top_k_actions = np.argsort(logits_np)[::-1][:K].tolist()
            g_noise       = np.zeros(CFG["action_dim"])

        q_values    = {}
        visit_count = {a: 0 for a in top_k_actions}
        candidates  = list(top_k_actions)

        phases         = max(1, int(np.log2(K))) if K > 1 else 1
        sims_per_phase = CFG["num_simulations"] // phases

        for phase in range(phases):
            if not candidates:
                break
            per_candidate = max(1, sims_per_phase // len(candidates))

            for action in candidates:
                for _ in range(per_candidate):
                    # Simulate action in world model (no real env step needed)
                    a_oh                    = F.one_hot(torch.tensor([action]),
                                              num_classes=CFG["action_dim"]).float().to(DEVICE)
                    next_s, pred_r          = model.dynamics(state, a_oh)
                    _, next_v               = model.predict(next_s)
                    q                       = pred_r.item() + disc * next_v.item()
                    n                       = visit_count[action]
                    q_values[action]        = (q_values.get(action, q) * n + q) / (n + 1)
                    visit_count[action]    += 1

            # Eliminate bottom half of candidates
            if phase < phases - 1 and len(candidates) > 1:
                max_v  = max(visit_count.values()) if visit_count else 1
                scored = sorted(
                    [(a, g_noise[a] + logits_np[a] + (50 + max_v) * 0.1 * q_values.get(a, value_np))
                     for a in candidates],
                    key=lambda x: x[1], reverse=True
                )
                candidates = [a for a, _ in scored[:max(1, len(candidates) // 2)]]

        best_action = max(q_values, key=q_values.get) if q_values else int(np.argmax(logits_np))

        # Improved policy: softmax over completed Q-values (σ-transform, paper eq. 19)
        max_v   = max(visit_count.values()) if visit_count else 1
        all_q   = np.array([q_values.get(a, value_np) for a in range(CFG["action_dim"])])
        all_q_t = (50 + max_v) * 0.1 * all_q
        all_q_t -= all_q_t.max()   # numerical stability
        imp_pol = np.exp(all_q_t)
        imp_pol /= imp_pol.sum()

    model.train()
    return best_action, imp_pol, value_np


# ════════════════════════════════════════════════════════════════════════════
# 5.  REPLAY BUFFER
# ════════════════════════════════════════════════════════════════════════════

Transition = collections.namedtuple(
    "Transition",
    ["obs", "action", "reward", "next_obs", "done",
     "policy_target", "value_target", "buffer_index"]
)


class ReplayBuffer:
    """
    Circular experience replay buffer.

    Why replay: sequential game steps are highly correlated.
    Sampling randomly from a buffer breaks these correlations
    and stabilises neural network training.
    """
    def __init__(self, max_size):
        self.buffer   = []
        self.max_size = max_size
        self.index    = 0

    def push(self, *args):
        t = Transition(*args, buffer_index=self.index)
        if len(self.buffer) < self.max_size:
            self.buffer.append(t)
        else:
            self.buffer[self.index % self.max_size] = t
        self.index += 1

    def sample(self, n):
        return random.sample(self.buffer, min(n, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


# ════════════════════════════════════════════════════════════════════════════
# 6.  MIXED VALUE TARGET (Section 4.3)
# ════════════════════════════════════════════════════════════════════════════

def compute_mixed_value_target(trans, model, training_step, total_buffer_index):
    """
    Use n-step TD for fresh/early data, SVE for stale data.

    Stale = collected by an old policy. n-step TD on stale data gives
    off-policy estimates. SVE re-runs the CURRENT model on the old state,
    giving a corrected estimate that reflects what the current policy
    would actually get. This is the off-policy correction in EZ-V2.
    """
    is_early = training_step < CFG["T1"]
    is_fresh = trans.buffer_index > (total_buffer_index - CFG["T2"])
    if is_early or is_fresh:
        return trans.value_target

    model.eval()
    with torch.no_grad():
        obs_t  = torch.FloatTensor(trans.obs).unsqueeze(0).to(DEVICE)
        _, val = model.predict(model.represent(obs_t))
        sve    = val.item()
    model.train()
    return sve


# ════════════════════════════════════════════════════════════════════════════
# 7.  LOSS FUNCTION (paper equation 3)
# ════════════════════════════════════════════════════════════════════════════

def cosine_loss(pred, target):
    """Negative cosine similarity — direction matters, not magnitude."""
    return -(F.normalize(pred, dim=-1) * F.normalize(target, dim=-1)).sum(dim=-1).mean()


def compute_loss(model, target_model, batch, training_step, total_buf_idx):
    obs_b      = torch.FloatTensor(np.array([t.obs           for t in batch])).to(DEVICE)
    act_b      = torch.LongTensor( np.array([t.action        for t in batch])).to(DEVICE)
    rew_b      = torch.FloatTensor(np.array([t.reward        for t in batch])).to(DEVICE)
    pol_b      = torch.FloatTensor(np.array([t.policy_target for t in batch])).to(DEVICE)
    nxt_obs_b  = torch.FloatTensor(np.array([t.next_obs      for t in batch])).to(DEVICE)
    val_tgt_b  = torch.FloatTensor([
        compute_mixed_value_target(t, model, training_step, total_buf_idx) for t in batch
    ]).to(DEVICE)

    state         = model.represent(obs_b)
    logits, value = model.predict(state)
    total_loss    = torch.tensor(0.0, device=DEVICE)

    for step in range(CFG["unroll_steps"]):
        act_oh             = F.one_hot(act_b, CFG["action_dim"]).float()
        next_state, pred_r = model.dynamics(state, act_oh)

        # L_R: reward prediction
        loss_r = F.mse_loss(pred_r, rew_b)

        # L_P: policy — cross-entropy vs improved policy from Gumbel search
        loss_p = F.cross_entropy(logits, pol_b)

        # L_V: value — MSE vs mixed value target
        loss_v = F.mse_loss(value, val_tgt_b)

        # L_G: temporal consistency — predicted future state vs encoded real one
        with torch.no_grad():
            tgt_proj = target_model.target_project(target_model.represent(nxt_obs_b))
        loss_g = cosine_loss(model.project(next_state, use_predictor=True), tgt_proj.detach())

        total_loss = total_loss + (
            CFG["coeff_reward"]      * loss_r +
            CFG["coeff_policy"]      * loss_p +
            CFG["coeff_value"]       * loss_v +
            CFG["coeff_consistency"] * loss_g
        ) / CFG["unroll_steps"]

        state         = next_state.detach()
        logits, value = model.predict(state)

    return total_loss, loss_r.item(), loss_p.item(), loss_v.item()


# ════════════════════════════════════════════════════════════════════════════
# 8.  TRAINING LOOP
# ════════════════════════════════════════════════════════════════════════════

def td_return(rewards, final_value, n, gamma):
    G = final_value
    for r in reversed(rewards[-n:]):
        G = r + gamma * G
    return G


def soft_update(model, target_model, tau=0.01):
    for p, p_t in zip(model.parameters(), target_model.parameters()):
        p_t.data.copy_(tau * p.data + (1 - tau) * p_t.data)


def evaluate(model, n_episodes=5):
    env   = gym.make(CFG["env_name"])
    total = 0.0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done   = False
        ep_r   = 0.0
        while not done:
            action, _, _ = gumbel_search(model, obs, 0, 999_999, use_noise=False)
            obs, r, te, tr, _ = env.step(action)
            done  = te or tr
            ep_r += r
        total += ep_r
    env.close()
    return total / n_episodes


def train():
    print("\n" + "="*60)
    print("  EfficientZero V2 — Simplified Training (Fixed)")
    print("="*60)

    model        = EZV2Model().to(DEVICE)
    target_model = EZV2Model().to(DEVICE)
    target_model.load_state_dict(model.state_dict())
    for p in target_model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    buffer = ReplayBuffer(CFG["buffer_size"])
    env    = gym.make(CFG["env_name"])

    log_file = open("results/training_log.csv", "w", newline="")
    writer   = csv.writer(log_file)
    writer.writerow(["env_step", "eval_reward", "loss", "loss_reward",
                     "loss_policy", "loss_value", "epsilon", "lr"])

    obs, _         = env.reset(seed=SEED)
    ep_rewards_buf = []
    training_step  = 0
    best_reward    = -float("inf")

    print(f"\n[Train] Starting. Buffer needs {CFG['min_buffer_size']} transitions before training.")
    print(f"[Train] Epsilon-greedy: {CFG['epsilon_start']} → {CFG['epsilon_end']} over {CFG['epsilon_decay']:,} steps\n")

    for env_step in range(1, CFG["total_steps"] + 1):
        epsilon = get_epsilon(env_step)

        # Action selection: random (exploration) or Gumbel search (planning)
        if len(buffer) < CFG["min_buffer_size"] or np.random.random() < epsilon:
            action        = env.action_space.sample()
            policy_target = np.ones(CFG["action_dim"]) / CFG["action_dim"]
            value_target  = 0.0
        else:
            action, policy_target, value_target = gumbel_search(
                model, obs, env_step, training_step, use_noise=True
            )

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        ep_rewards_buf.append(reward)

        # n-step TD value target
        if len(ep_rewards_buf) >= CFG["td_steps"] or done:
            if done:
                final_val = 0.0
            else:
                with torch.no_grad():
                    obs_t     = torch.FloatTensor(next_obs).unsqueeze(0).to(DEVICE)
                    _, fv     = model.predict(model.represent(obs_t))
                    final_val = fv.item()
            td_tgt = td_return(ep_rewards_buf, final_val, CFG["td_steps"], CFG["discount"])
        else:
            td_tgt = value_target

        buffer.push(obs, action, reward, next_obs, done, policy_target, td_tgt)
        obs = next_obs

        if done:
            obs, _         = env.reset()
            ep_rewards_buf = []

        # Train
        if len(buffer) >= CFG["min_buffer_size"]:
            batch = buffer.sample(CFG["batch_size"])
            optimizer.zero_grad()
            loss, lr_l, lp, lv = compute_loss(model, target_model, batch,
                                               training_step, buffer.index)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            optimizer.step()
            soft_update(model, target_model)
            training_step += 1

            if env_step % CFG["eval_interval"] == 0:
                mean_reward = evaluate(model, CFG["eval_episodes"])
                cur_lr      = optimizer.param_groups[0]["lr"]
                print(f"[Step {env_step:>6,}]  reward={mean_reward:>6.1f}  "
                      f"loss={loss.item():.4f}  ε={epsilon:.3f}  lr={cur_lr:.2e}")
                writer.writerow([env_step, mean_reward, loss.item(), lr_l, lp, lv,
                                 epsilon, cur_lr])
                log_file.flush()
                scheduler.step(mean_reward)

                if mean_reward > best_reward:
                    best_reward = mean_reward
                    torch.save({"model_state": model.state_dict(),
                                "env_step": env_step, "eval_reward": mean_reward,
                                "config": CFG}, "checkpoints/model_best.pt")
                    print(f"           ★ New best: {mean_reward:.1f}")

            if env_step % CFG["save_interval"] == 0:
                torch.save({"model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "env_step": env_step, "training_step": training_step,
                            "config": CFG},
                           f"checkpoints/model_{env_step:06d}.pt")

    env.close()
    log_file.close()
    torch.save({"model_state": model.state_dict(), "env_step": CFG["total_steps"],
                "training_step": training_step, "config": CFG},
               "checkpoints/model_final.pt")

    print(f"\n[Done] Training complete. Best reward: {best_reward:.1f}")
    print("[Done] Best model  → checkpoints/model_best.pt   ← use THIS for test.py")
    print("[Done] Final model → checkpoints/model_final.pt")


if __name__ == "__main__":
    train()