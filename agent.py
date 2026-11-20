import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np


class RunningMeanStd:
    """Tracks running mean and variance using Welford's algorithm with outlier clipping."""
    def __init__(self, shape=(), epsilon=1e-4, clip_range=10.0):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
        self.clip_range = clip_range

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if len(x.shape) > 1 else 1

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x):
        norm_x = (x - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(norm_x, -self.clip_range, self.clip_range)


class ReplayBuffer:
    """High-performance circular replay buffer stored in NumPy arrays."""
    def __init__(self, obs_dim, act_dim, capacity=1_000_000):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def store(self, obs, act, rew, next_obs, done):
        self.obs[self.ptr] = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr] = act
        self.rewards[self.ptr] = rew
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.obs[idx], dtype=torch.float32),
            torch.as_tensor(self.actions[idx], dtype=torch.float32),
            torch.as_tensor(self.rewards[idx], dtype=torch.float32),
            torch.as_tensor(self.next_obs[idx], dtype=torch.float32),
            torch.as_tensor(self.dones[idx], dtype=torch.float32),
        )

    def save(self, filepath):
        """Save replay buffer contents to fast uncompressed .npz archive."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savez(
            filepath,
            obs=self.obs[:self.size],
            next_obs=self.next_obs[:self.size],
            actions=self.actions[:self.size],
            rewards=self.rewards[:self.size],
            dones=self.dones[:self.size],
            ptr=self.ptr,
            size=self.size
        )
        print(f"[BUFFER] Saved {self.size:,} transitions to {filepath}")

    def load(self, filepath):
        """Load replay buffer contents from .npz archive."""
        if not os.path.exists(filepath):
            print(f"[BUFFER] No buffer found at {filepath}, starting empty.")
            return False
        try:
            data = np.load(filepath)
            n = int(data["size"])
            self.obs[:n] = data["obs"]
            self.next_obs[:n] = data["next_obs"]
            self.actions[:n] = data["actions"]
            self.rewards[:n] = data["rewards"]
            self.dones[:n] = data["dones"]
            self.size = n
            self.ptr = int(data["ptr"]) % self.capacity
            print(f"[BUFFER] Successfully loaded {self.size:,} transitions from {filepath}")
            return True
        except Exception as e:
            print(f"[BUFFER] Warning: Failed to load buffer from {filepath} ({e}). Starting fresh.")
            return False


class SquashedGaussianActor(nn.Module):
    """Gaussian Actor with numerically stable Tanh squashing outputting in [-1, 1]."""
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.act_dim = act_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(hidden_dim, act_dim)
        self.log_std_layer = nn.Linear(hidden_dim, act_dim)

    def forward(self, obs, deterministic=False, with_logprob=True):
        feat = self.net(obs)
        mu = self.mu_layer(feat)
        log_std = torch.clamp(self.log_std_layer(feat), -20, 2)
        std = torch.exp(log_std)

        dist = Normal(mu, std)
        if deterministic:
            u = mu
        else:
            u = dist.rsample()

        action = torch.tanh(u)

        if with_logprob:
            # Numerically stable log determinant for Tanh squashing (OpenAI Spinning Up formula)
            logp = dist.log_prob(u).sum(dim=-1, keepdim=True)
            logp -= (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(dim=-1, keepdim=True)
        else:
            logp = None

        return action, logp


class QCritic(nn.Module):
    """Twin Q-network with LayerNorm for value stabilization."""
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs, act):
        sa = torch.cat([obs, act], dim=-1)
        return self.q1(sa), self.q2(sa)


class SACAgent:
    """High-Quality Soft Actor-Critic (SAC) Agent for Humanoid-v5."""
    def __init__(
        self,
        obs_dim=348,
        act_dim=17,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        auto_alpha=True,
        device="cpu",
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device(device)

        # Networks with LayerNorm
        self.actor = SquashedGaussianActor(obs_dim, act_dim, hidden_dim).to(self.device)
        self.critic = QCritic(obs_dim, act_dim, hidden_dim).to(self.device)
        self.critic_target = QCritic(obs_dim, act_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # Automatic Entropy Tuning
        self.auto_alpha = auto_alpha
        if self.auto_alpha:
            self.target_entropy = -float(act_dim)
            # Initialize log_alpha to log(0.2) ~ -1.609
            self.log_alpha = torch.tensor([-1.609], requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha

        # Observation Normalizer with outlier clipping
        self.normalizer = RunningMeanStd(shape=(obs_dim,), clip_range=10.0)

    def select_action(self, obs, deterministic=False, update_normalizer=True):
        if update_normalizer:
            self.normalizer.update(obs)
        norm_obs = self.normalizer.normalize(obs)

        with torch.no_grad():
            obs_t = torch.as_tensor(norm_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            action, _ = self.actor(obs_t, deterministic=deterministic, with_logprob=False)
            return action.cpu().numpy()[0]

    def update(self, batch):
        obs, act, rew, next_obs, dones = [x.to(self.device) for x in batch]

        # 1. Update Critic
        with torch.no_grad():
            next_action, next_logp = self.actor(next_obs)
            target_q1, target_q2 = self.critic_target(next_obs, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_logp
            backup = rew + self.gamma * (1.0 - dones) * target_q

        current_q1, current_q2 = self.critic(obs, act)
        critic_loss = F.mse_loss(current_q1, backup) + F.mse_loss(current_q2, backup)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
        self.critic_optimizer.step()

        # 2. Update Actor
        pred_action, logp = self.actor(obs)
        q1_pi, q2_pi = self.critic(obs, pred_action)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * logp - q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        self.actor_optimizer.step()

        # 3. Update Alpha (Entropy temperature with safety clamping)
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            # Safety clamp: alpha stays strictly between [0.028, 0.35] (protects bipedal elasticity)
            self.log_alpha.data.clamp_(min=-3.57, max=-1.05)
            self.alpha = self.log_alpha.exp().item()

        # 4. Soft Target Update
        with torch.no_grad():
            for p, p_target in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_target.data.mul_(1.0 - self.tau)
                p_target.data.add_(self.tau * p.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha,
        }

    def save(self, filepath, step, episode, best_reward):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            "step": step,
            "episode": episode,
            "best_reward": best_reward,
            "actor_state": self.actor.state_dict(),
            "critic_state": self.critic.state_dict(),
            "critic_target_state": self.critic_target.state_dict(),
            "actor_opt_state": self.actor_optimizer.state_dict(),
            "critic_opt_state": self.critic_optimizer.state_dict(),
            "normalizer_mean": self.normalizer.mean,
            "normalizer_var": self.normalizer.var,
            "normalizer_count": self.normalizer.count,
        }
        if self.auto_alpha:
            checkpoint["log_alpha"] = self.log_alpha.data
            checkpoint["alpha_opt_state"] = self.alpha_optimizer.state_dict()

        torch.save(checkpoint, filepath)
        print(f"[SAVE] Checkpoint saved successfully to {filepath} at step {step:,}")

    def load(self, filepath):
        if not os.path.exists(filepath):
            return 0, 0, -float("inf")

        try:
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(filepath, map_location=self.device)

        self.actor.load_state_dict(checkpoint["actor_state"])
        self.critic.load_state_dict(checkpoint["critic_state"])
        self.critic_target.load_state_dict(checkpoint["critic_target_state"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_opt_state"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_opt_state"])

        self.normalizer.mean = checkpoint["normalizer_mean"]
        self.normalizer.var = checkpoint["normalizer_var"]
        self.normalizer.count = checkpoint["normalizer_count"]

        if self.auto_alpha and "log_alpha" in checkpoint:
            self.log_alpha.data = checkpoint["log_alpha"]
            self.alpha_optimizer.load_state_dict(checkpoint["alpha_opt_state"])
            self.alpha = self.log_alpha.exp().item()

        step = checkpoint.get("step", 0)
        episode = checkpoint.get("episode", 0)
        best_reward = checkpoint.get("best_reward", -float("inf"))

        print(f"[LOAD] Checkpoint loaded: Step {step:,}, Episode {episode:,}, Best Score: {best_reward:.1f}")
        return step, episode, best_reward
