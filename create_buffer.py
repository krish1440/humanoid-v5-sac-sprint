import os
import sys
import time
import numpy as np
import gymnasium as gym
from agent import SACAgent, ReplayBuffer


def main():
    NUM_WORKERS = 8
    TARGET_TRANSITIONS = 100_000
    checkpoint_dir = "checkpoints"
    latest_checkpoint = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
    buffer_save_path = os.path.join(checkpoint_dir, "replay_buffer.npz")
    buffer_backup_path = os.path.join(checkpoint_dir, "backup_1m_milestone", "replay_buffer.npz")

    if not os.path.exists(latest_checkpoint):
        print(f"[ERROR] Checkpoint not found at {latest_checkpoint}!")
        sys.exit(1)

    print("=" * 88)
    print(" GENERATING 100,000 EXPERT REPLAY BUFFER FROM 1M MODEL")
    print(f" Target: {TARGET_TRANSITIONS:,} transitions across {NUM_WORKERS} parallel environments")
    print(" Estimated duration: ~60-80 seconds")
    print("=" * 88)

    # 1. Initialize environments
    envs = []
    for _ in range(NUM_WORKERS):
        raw_env = gym.make("Humanoid-v5", render_mode=None)
        scaled_env = gym.wrappers.RescaleAction(raw_env, min_action=-1.0, max_action=1.0)
        envs.append(scaled_env)

    obs_dim = envs[0].observation_space.shape[0]
    act_dim = envs[0].action_space.shape[0]

    # 2. Initialize Agent & Load Checkpoint
    agent = SACAgent(obs_dim=obs_dim, act_dim=act_dim)
    agent.load(latest_checkpoint)
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, capacity=TARGET_TRANSITIONS + 10_000)

    # 3. Reset Workers
    worker_obs = np.zeros((NUM_WORKERS, obs_dim), dtype=np.float32)
    worker_scores = np.zeros(NUM_WORKERS)
    worker_lens = np.zeros(NUM_WORKERS, dtype=int)
    for i in range(NUM_WORKERS):
        worker_obs[i], _ = envs[i].reset()

    # 4. Collection loop
    t0 = time.time()
    steps_collected = 0
    completed_episodes = 0
    ep_scores_list = []

    while steps_collected < TARGET_TRANSITIONS:
        # Select actions with trained 1M policy (no weight updates!)
        actions = agent.select_action(worker_obs, deterministic=False, update_normalizer=False)

        for i in range(NUM_WORKERS):
            act = actions[i]
            cur_obs = worker_obs[i].copy()
            next_obs, rew, terminated, truncated, _ = envs[i].step(act)
            done = terminated or truncated

            # Store normalized transitions
            norm_cur = agent.normalizer.normalize(cur_obs)
            norm_next = agent.normalizer.normalize(next_obs)
            replay_buffer.store(norm_cur, act, rew, norm_next, float(terminated))

            worker_scores[i] += rew
            worker_lens[i] += 1
            steps_collected += 1

            if done:
                completed_episodes += 1
                ep_scores_list.append(worker_scores[i])
                worker_obs[i], _ = envs[i].reset()
                worker_scores[i] = 0.0
                worker_lens[i] = 0
            else:
                worker_obs[i] = next_obs

            if steps_collected >= TARGET_TRANSITIONS:
                break

        if steps_collected % 50_000 == 0:
            elapsed = time.time() - t0
            sps = steps_collected / (elapsed + 1e-6)
            avg_sc = np.mean(ep_scores_list[-10:]) if ep_scores_list else 0.0
            print(f" Collected: {steps_collected:9,d} / {TARGET_TRANSITIONS:,} transitions ({steps_collected/TARGET_TRANSITIONS:5.1%}) | "
                  f"Speed: {sps:5.0f} steps/sec | Avg Score: {avg_sc:6.1f}")

    elapsed_total = time.time() - t0
    print("-" * 88)
    print(f" [SUCCESS] Collection finished in {elapsed_total:.1f}s ({steps_collected/elapsed_total:.0f} steps/sec)!")
    print(f" Total completed episodes: {completed_episodes} | Overall Average Score: {np.mean(ep_scores_list):.1f}")

    # 5. Save buffer to checkpoints/replay_buffer.npz
    replay_buffer.save(buffer_save_path)

    # 6. Take backup of the buffer into checkpoints/backup_1m_milestone/
    replay_buffer.save(buffer_backup_path)
    print(f" [BACKUP] Golden buffer archive saved to: {buffer_backup_path}")
    print("=" * 88)

    for e in envs:
        e.close()


if __name__ == "__main__":
    main()
