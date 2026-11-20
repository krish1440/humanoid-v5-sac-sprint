import os
import sys
import time
import csv
from collections import deque
import numpy as np
import pygame
import gymnasium as gym
from agent import SACAgent, ReplayBuffer


def create_hud_surface(width, height, stats, is_fast, is_paused, save_banner_timer, num_workers=8):
    """Draws an elegant top HUD overlay with worker status."""
    hud = pygame.Surface((width, height), pygame.SRCALPHA)

    header_height = 72
    pygame.draw.rect(hud, (15, 20, 30, 230), (0, 0, width, header_height))
    pygame.draw.line(hud, (0, 220, 180, 200), (0, header_height), (width, header_height), 2)

    font_large = pygame.font.SysFont("Consolas", 16, bold=True)
    font_med = pygame.font.SysFont("Consolas", 13)
    font_bold = pygame.font.SysFont("Consolas", 13, bold=True)

    # --- ROW 1 (y = 10) ---
    title = font_large.render(f"HUMANOID-v5 [PARALLEL {num_workers}x SPRINT]", True, (0, 255, 200))
    hud.blit(title, (15, 10))

    ctrl_text = "[SPACE] Pause/Resume   [V] Turbo / Visual   [S] Save   [ESC] Exit"
    ctrl_surf = font_bold.render(ctrl_text, True, (240, 240, 240))
    ctrl_x = (width - ctrl_surf.get_width()) // 2
    hud.blit(ctrl_surf, (ctrl_x, 12))

    if is_paused:
        status_text = "|| PAUSED"
        status_color = (255, 180, 0)
    elif is_fast:
        status_text = f">> TURBO PARALLEL ({num_workers} ENVS)"
        status_color = (50, 255, 120)
    else:
        status_text = "> LIVE 3D WORKER-0 (50 FPS)"
        status_color = (120, 200, 255)

    status_surf = font_bold.render(status_text, True, status_color)
    hud.blit(status_surf, (width - status_surf.get_width() - 15, 12))

    # --- ROW 2 (y = 42) ---
    step_str = f"Step: {stats['step']:,}  |  Ep: {stats['episode']:,}  |  Last Ep: {stats['ep_reward']:.1f}  |  Best: {stats['best_reward']:.1f}"
    hud.blit(font_med.render(step_str, True, (255, 215, 0)), (15, 42))

    torso_z_str = f"W0 Height: {stats['torso_z']:.2f} m"
    fwd_vel_str = f"W0 Fwd Vel: {stats['fwd_vel']:+.2f} m/s"
    z_color = (50, 255, 120) if (1.0 <= stats['torso_z'] <= 2.0) else (255, 70, 70)

    kinematics_str = f"{torso_z_str}   |   {fwd_vel_str}"
    kinematics_surf = font_med.render(kinematics_str, True, z_color)
    hud.blit(kinematics_surf, (width - kinematics_surf.get_width() - 15, 42))

    if save_banner_timer > 0:
        banner = font_bold.render("*** CHECKPOINT & BUFFER SAVED ***", True, (50, 255, 120))
        banner_rect = banner.get_rect(center=(width // 2, header_height + 25))
        pygame.draw.rect(hud, (0, 50, 25, 230), banner_rect.inflate(26, 10), border_radius=6)
        pygame.draw.rect(hud, (50, 255, 120), banner_rect.inflate(26, 10), width=1, border_radius=6)
        hud.blit(banner, banner_rect)

    return hud


def draw_turbo_card(screen, width, height, stats, last_losses, num_workers=8, steps_per_sec=0.0, buffer_size=0):
    """Draws a clean telemetry dashboard card in TURBO parallel mode."""
    font_title = pygame.font.SysFont("Consolas", 18, bold=True)
    font_info = pygame.font.SysFont("Consolas", 13)
    font_metric = pygame.font.SysFont("Consolas", 14, bold=True)

    card_w, card_h = 660, 270
    card_x = (width - card_w) // 2
    card_y = (height - card_h) // 2 + 20

    card_rect = pygame.Rect(card_x, card_y, card_w, card_h)
    pygame.draw.rect(screen, (20, 28, 42), card_rect, border_radius=12)
    pygame.draw.rect(screen, (0, 220, 180), card_rect, width=2, border_radius=12)

    t_surf = font_title.render(f"⚡ HIGH-THROUGHPUT SPRINT TRAINING ({num_workers} ENVS) ⚡", True, (50, 255, 180))
    screen.blit(t_surf, (card_x + (card_w - t_surf.get_width()) // 2, card_y + 16))

    sub_surf = font_info.render(f"Optimizing forward velocity toward 6,500+ sprint milestone", True, (170, 190, 210))
    screen.blit(sub_surf, (card_x + (card_w - sub_surf.get_width()) // 2, card_y + 44))

    speed_surf = font_metric.render(f"Speed: {steps_per_sec:.0f} steps/second  |  Buffer RAM: {buffer_size:,} transitions", True, (255, 215, 0))
    screen.blit(speed_surf, (card_x + (card_w - speed_surf.get_width()) // 2, card_y + 68))

    pygame.draw.line(screen, (40, 55, 80), (card_x + 30, card_y + 96), (card_x + card_w - 30, card_y + 96), 1)

    # Grid Metrics (2 columns)
    m_left_x = card_x + 40
    m_right_x = card_x + card_w // 2 + 20
    row_y = card_y + 112

    # Column 1
    m1 = font_metric.render(f"Total Steps:   {stats['step']:,}", True, (240, 240, 240))
    screen.blit(m1, (m_left_x, row_y))

    m2 = font_metric.render(f"Total Episodes:{stats['episode']:,}", True, (240, 240, 240))
    screen.blit(m2, (m_left_x, row_y + 26))

    m3 = font_metric.render(f"Best Reward:   {stats['best_reward']:.1f}", True, (50, 255, 120))
    screen.blit(m3, (m_left_x, row_y + 52))

    m4 = font_metric.render(f"Last Ep Score: {stats['ep_reward']:.1f}", True, (255, 215, 0))
    screen.blit(m4, (m_left_x, row_y + 78))

    # Column 2
    l1 = font_metric.render(f"Critic Loss:   {last_losses['critic_loss']:.3f}", True, (220, 220, 220))
    screen.blit(l1, (m_right_x, row_y))

    l2 = font_metric.render(f"Actor Loss:    {last_losses['actor_loss']:.3f}", True, (220, 220, 220))
    screen.blit(l2, (m_right_x, row_y + 26))

    l3 = font_metric.render(f"Entropy Temp α:{last_losses['alpha']:.4f}", True, (220, 220, 220))
    screen.blit(l3, (m_right_x, row_y + 52))

    l4 = font_metric.render(f"Active Workers:{num_workers} Cores", True, (0, 220, 255))
    screen.blit(l4, (m_right_x, row_y + 78))

    # Footer
    f_surf = font_info.render("Official Gymnasium Humanoid-v5 Default Rewards | Checkpoint & Buffer auto-saved", True, (130, 150, 175))
    screen.blit(f_surf, (card_x + (card_w - f_surf.get_width()) // 2, card_y + card_h - 26))


def main():
    # 1. Configuration
    NUM_WORKERS = 8
    win_w, win_h = 1024, 768
    render_w, render_h = win_w, win_h

    # 2. Pygame UI Initialization
    pygame.init()
    pygame.display.set_caption(f"MuJoCo Humanoid-v5 SAC - Parallel ({NUM_WORKERS} Workers)")
    screen = pygame.display.set_mode((win_w, win_h), pygame.DOUBLEBUF)
    clock = pygame.time.Clock()

    # 3. Create Parallel Environments
    envs = []
    for i in range(NUM_WORKERS):
        r_mode = "rgb_array" if i == 0 else None
        raw_env = gym.make("Humanoid-v5", render_mode=r_mode, width=render_w, height=render_h)
        scaled_env = gym.wrappers.RescaleAction(raw_env, min_action=-1.0, max_action=1.0)
        envs.append(scaled_env)

    obs_dim = envs[0].observation_space.shape[0]
    act_dim = envs[0].action_space.shape[0]

    # 4. Agent and Replay Buffer (Fine-tuning rate for stable sprint scaling)
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_checkpoint = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
    best_checkpoint = os.path.join(checkpoint_dir, "best_model.pt")
    buffer_file = os.path.join(checkpoint_dir, "replay_buffer.npz")

    agent = SACAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=256,
        lr=1e-4,  # Fine-tuning sprint learning rate
        gamma=0.99,
        tau=0.005,
        auto_alpha=True,
    )
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, capacity=1_000_000)

    # Load 1M Weights
    start_step, start_episode, best_reward = agent.load(latest_checkpoint)
    total_steps = start_step
    episode_count = start_episode

    # Load Golden Replay Buffer from Disk
    replay_buffer.load(buffer_file)

    # State variables
    is_fast = True
    is_paused = False
    save_banner_timer = 0
    batch_size = 256
    save_freq_steps = 25_000

    # Logging Setup
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "training_log.csv")
    if not os.path.exists(log_file):
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "total_steps", "worker_id", "ep_reward", "ep_length", "mean_20", "best_reward", "avg_fwd_vel", "critic_loss", "actor_loss", "alpha"])

    recent_returns = deque(maxlen=20)
    last_losses = {"critic_loss": 0.0, "actor_loss": 0.0, "alpha": agent.alpha}

    # Worker tracking
    worker_obs = np.zeros((NUM_WORKERS, obs_dim), dtype=np.float32)
    worker_ep_rewards = np.zeros(NUM_WORKERS, dtype=np.float64)
    worker_ep_lens = np.zeros(NUM_WORKERS, dtype=np.int32)
    worker_fwd_vels = [[] for _ in range(NUM_WORKERS)]

    for i in range(NUM_WORKERS):
        reset_obs, _ = envs[i].reset()
        worker_obs[i] = reset_obs

    stats = {
        "step": total_steps,
        "episode": episode_count,
        "ep_reward": 0.0,
        "best_reward": best_reward if best_reward != -float("inf") else 0.0,
        "torso_z": float(worker_obs[0, 0]),
        "fwd_vel": 0.0,
    }

    print("=" * 92)
    print(f" HUMANOID-v5 PARALLEL ({NUM_WORKERS} WORKERS) SPRINT TRAINING INITIALIZED")
    print(" Controls: [SPACE] Pause | [V] Turbo/Live 3D | [S] Instant Save | [ESC] Exit")
    print(f" Resumed from LATEST CHECKPOINT at Step: {total_steps:,} | Ep: {episode_count:,} | Best Score: {stats['best_reward']:.1f}")
    print(f" Replay Buffer pre-loaded from disk: {replay_buffer.size:,} expert transitions in RAM!")
    print(" Target Sprint Score: 6,500+ | All rewards 100% official Gymnasium defaults.")
    print("=" * 92)

    speed_timer = time.time()
    speed_steps = 0
    steps_per_sec = 0.0
    render_step_counter = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("\n[EXIT] Window closed by user. Saving checkpoint and buffer...")
                agent.save(latest_checkpoint, total_steps, episode_count, stats["best_reward"])
                replay_buffer.save(buffer_file)
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    is_paused = not is_paused
                    print(f"[HOTKEY] Spacebar pressed -> Paused: {is_paused}")
                elif event.key == pygame.K_v:
                    is_fast = not is_fast
                    mode_name = f"TURBO PARALLEL ({NUM_WORKERS} Envs)" if is_fast else "LIVE 3D (Worker 0)"
                    print(f"[HOTKEY] 'V' pressed -> Mode: {mode_name}")
                elif event.key == pygame.K_s:
                    agent.save(latest_checkpoint, total_steps, episode_count, stats["best_reward"])
                    replay_buffer.save(buffer_file)
                    save_banner_timer = 90
                    print(f"[HOTKEY] 'S' pressed -> Instant Checkpoint & Buffer Saved at step {total_steps:,}")
                elif event.key == pygame.K_ESCAPE:
                    print("\n[EXIT] ESC pressed. Saving checkpoint and buffer...")
                    agent.save(latest_checkpoint, total_steps, episode_count, stats["best_reward"])
                    replay_buffer.save(buffer_file)
                    running = False

        if not running:
            break

        if is_paused:
            hud_surf = create_hud_surface(win_w, win_h, stats, is_fast, is_paused, save_banner_timer, NUM_WORKERS)
            screen.blit(hud_surf, (0, 0))
            pygame.display.flip()
            clock.tick(30)
            continue

        # Pure Vectorized Action Selection
        actions = agent.select_action(worker_obs, deterministic=False, update_normalizer=False)

        # Step all workers
        for i in range(NUM_WORKERS):
            act = actions[i]
            cur_obs = worker_obs[i].copy()
            next_obs, rew, terminated, truncated, info = envs[i].step(act)
            done = terminated or truncated

            # Store normalized transitions in Replay Buffer
            norm_cur = agent.normalizer.normalize(cur_obs)
            norm_next = agent.normalizer.normalize(next_obs)
            replay_buffer.store(norm_cur, act, rew, norm_next, float(terminated))

            worker_ep_rewards[i] += float(rew)
            worker_ep_lens[i] += 1
            fwd_v = float(info.get("x_velocity", 0.0))
            worker_fwd_vels[i].append(fwd_v)

            total_steps += 1
            speed_steps += 1

            # Checkpoint autosave frequency
            if total_steps % save_freq_steps == 0:
                agent.save(latest_checkpoint, total_steps, episode_count, stats["best_reward"])
                replay_buffer.save(buffer_file)
                save_banner_timer = 90
                print(f"[SAVE] Checkpoint & Buffer saved successfully at step {total_steps:,}")

            if done:
                episode_count += 1
                completed_score = worker_ep_rewards[i]
                completed_len = worker_ep_lens[i]
                avg_fwd = float(np.mean(worker_fwd_vels[i])) if worker_fwd_vels[i] else 0.0
                recent_returns.append(completed_score)
                mean_20 = float(np.mean(recent_returns))

                stats["episode"] = episode_count
                stats["ep_reward"] = completed_score

                # Save new all-time best
                if completed_score > stats["best_reward"]:
                    stats["best_reward"] = completed_score
                    agent.save(best_checkpoint, total_steps, episode_count, completed_score)
                    print(f"       >>> 🔥 NEW BEST SCORE: {completed_score:.1f} (Worker W{i}) | Saved Checkpoint! <<<")

                # Log to CSV with exact worker_id
                with open(log_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        episode_count,
                        total_steps,
                        f"W{i}",
                        round(completed_score, 1),
                        completed_len,
                        round(mean_20, 1),
                        round(stats["best_reward"], 1),
                        round(avg_fwd, 2),
                        round(last_losses["critic_loss"], 3),
                        round(last_losses["actor_loss"], 3),
                        round(last_losses["alpha"], 4)
                    ])

                print(f"[EP {episode_count:5d}] Steps: {total_steps:10,d} | Score: {completed_score:7.1f} | "
                      f"Len: {completed_len:4d} | Mean(20): {mean_20:7.1f} | Best: {stats['best_reward']:7.1f} | "
                      f"Fwd Vel: {avg_fwd:+.2f} m/s | α: {last_losses['alpha']:.3f} | "
                      f"Crit Loss: {last_losses['critic_loss']:7.3f} | Act Loss: {last_losses['actor_loss']:7.3f} | W{i}")

                # Reset only this finished worker
                reset_obs, _ = envs[i].reset()
                worker_obs[i] = reset_obs
                worker_ep_rewards[i] = 0.0
                worker_ep_lens[i] = 0
                worker_fwd_vels[i] = []
            else:
                worker_obs[i] = next_obs

        # Update telemetry stats from Worker 0
        stats["step"] = total_steps
        stats["torso_z"] = float(worker_obs[0, 0])
        stats["fwd_vel"] = float(worker_fwd_vels[0][-1]) if worker_fwd_vels[0] else 0.0

        # Gradient update step (Sampling from the pre-loaded 100k expert buffer)
        if replay_buffer.size >= batch_size:
            batch = replay_buffer.sample(batch_size)
            last_losses = agent.update(batch)

        # Measure steps/second
        now = time.time()
        if now - speed_timer >= 1.0:
            steps_per_sec = speed_steps / (now - speed_timer)
            speed_steps = 0
            speed_timer = now

        # Rendering
        render_step_counter += 1
        if is_fast:
            if render_step_counter % 15 == 0:
                screen.fill((12, 16, 24))
                draw_turbo_card(screen, win_w, win_h, stats, last_losses, NUM_WORKERS, steps_per_sec, replay_buffer.size)
                hud_surf = create_hud_surface(win_w, win_h, stats, is_fast, is_paused, save_banner_timer, NUM_WORKERS)
                screen.blit(hud_surf, (0, 0))
                pygame.display.flip()
        else:
            frame = envs[0].render()
            if frame is not None:
                surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
                scaled_surf = pygame.transform.scale(surf, (win_w, win_h))
                screen.blit(scaled_surf, (0, 0))
            hud_surf = create_hud_surface(win_w, win_h, stats, is_fast, is_paused, save_banner_timer, NUM_WORKERS)
            screen.blit(hud_surf, (0, 0))
            pygame.display.flip()
            clock.tick(50)

        if save_banner_timer > 0:
            save_banner_timer -= 1

    # Cleanup
    for e in envs:
        e.close()
    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
