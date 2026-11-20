import os
import sys
import time
import csv
from collections import deque
import numpy as np
import pygame
import gymnasium as gym
from agent import SACAgent, ReplayBuffer


def create_hud_surface(width, height, stats, is_fast, is_paused, save_banner_timer):
    """Draws an elegant, consolidated top HUD overlay with zero obstruction at the bottom."""
    hud = pygame.Surface((width, height), pygame.SRCALPHA)

    # Top Header Bar (Height = 72px)
    header_height = 72
    pygame.draw.rect(hud, (15, 20, 30, 230), (0, 0, width, header_height))
    pygame.draw.line(hud, (0, 180, 220, 180), (0, header_height), (width, header_height), 2)

    font_large = pygame.font.SysFont("Consolas", 16, bold=True)
    font_med = pygame.font.SysFont("Consolas", 13)
    font_bold = pygame.font.SysFont("Consolas", 13, bold=True)

    # --- ROW 1 (y = 10) ---
    # Title
    title = font_large.render("HUMANOID-v5", True, (0, 220, 255))
    hud.blit(title, (15, 10))

    # Controls Bar (Clean Centered Spacing)
    ctrl_text = "[SPACE] Pause/Resume   [V] Turbo / Live 3D   [S] Save   [ESC] Exit"
    ctrl_surf = font_bold.render(ctrl_text, True, (240, 240, 240))
    ctrl_x = (width - ctrl_surf.get_width()) // 2
    hud.blit(ctrl_surf, (ctrl_x, 12))

    # Status Tag (Top Right)
    if is_paused:
        status_text = "|| PAUSED"
        status_color = (255, 180, 0)
    elif is_fast:
        status_text = ">> TURBO (MAX SPEED)"
        status_color = (50, 255, 120)
    else:
        status_text = "> LIVE 3D (50 FPS)"
        status_color = (120, 200, 255)

    status_surf = font_bold.render(status_text, True, status_color)
    hud.blit(status_surf, (width - status_surf.get_width() - 15, 12))

    # --- ROW 2 (y = 42) ---
    # Left: Training Steps & Rewards
    step_str = f"Step: {stats['step']:,}  |  Ep: {stats['episode']:,}  |  Ep Reward: {stats['ep_reward']:.1f}  |  Best: {stats['best_reward']:.1f}"
    hud.blit(font_med.render(step_str, True, (255, 215, 0)), (15, 42))

    # Right: Robot Kinematics (Height & Speed)
    torso_z_str = f"Torso Height (z): {stats['torso_z']:.2f} m"
    fwd_vel_str = f"Forward Vel (vx): {stats['fwd_vel']:+.2f} m/s"
    z_color = (50, 255, 120) if (1.0 <= stats['torso_z'] <= 2.0) else (255, 70, 70)

    kinematics_str = f"{torso_z_str}   |   {fwd_vel_str}"
    kinematics_surf = font_med.render(kinematics_str, True, z_color)
    hud.blit(kinematics_surf, (width - kinematics_surf.get_width() - 15, 42))

    # Save Confirmation Banner (Pops below header)
    if save_banner_timer > 0:
        banner = font_bold.render("*** CHECKPOINT SAVED TO DISK ***", True, (50, 255, 120))
        banner_rect = banner.get_rect(center=(width // 2, header_height + 25))
        pygame.draw.rect(hud, (0, 50, 25, 230), banner_rect.inflate(26, 10), border_radius=6)
        pygame.draw.rect(hud, (50, 255, 120), banner_rect.inflate(26, 10), width=1, border_radius=6)
        hud.blit(banner, banner_rect)

    return hud


def draw_turbo_card(screen, width, height, stats, last_losses):
    """Draws a clean telemetry dashboard card in TURBO mode so the window stays informative and fluid."""
    font_title = pygame.font.SysFont("Consolas", 18, bold=True)
    font_info = pygame.font.SysFont("Consolas", 13)
    font_metric = pygame.font.SysFont("Consolas", 14, bold=True)

    card_w, card_h = 640, 260
    card_x = (width - card_w) // 2
    card_y = (height - card_h) // 2 + 20

    card_rect = pygame.Rect(card_x, card_y, card_w, card_h)
    pygame.draw.rect(screen, (20, 28, 42), card_rect, border_radius=12)
    pygame.draw.rect(screen, (0, 180, 220), card_rect, width=2, border_radius=12)

    # Title Banner
    t_surf = font_title.render("⚡ TURBO TRAINING MODE ACTIVE ⚡", True, (50, 255, 120))
    screen.blit(t_surf, (card_x + (card_w - t_surf.get_width()) // 2, card_y + 18))

    sub_surf = font_info.render("3D rendering bypassed to maximize learning throughput (~200+ steps/sec)", True, (170, 190, 210))
    screen.blit(sub_surf, (card_x + (card_w - sub_surf.get_width()) // 2, card_y + 48))

    sub2_surf = font_metric.render("Press [V] at any time to toggle Live 3D View", True, (255, 215, 0))
    screen.blit(sub2_surf, (card_x + (card_w - sub2_surf.get_width()) // 2, card_y + 72))

    pygame.draw.line(screen, (40, 55, 80), (card_x + 30, card_y + 100), (card_x + card_w - 30, card_y + 100), 1)

    # Telemetry Columns
    col1_x = card_x + 45
    col2_x = card_x + 340
    y_start = card_y + 115
    line_h = 28

    screen.blit(font_info.render(f"Step: {stats['step']:,}", True, (240, 240, 240)), (col1_x, y_start))
    screen.blit(font_info.render(f"Episode: {stats['episode']:,}", True, (240, 240, 240)), (col1_x, y_start + line_h))
    screen.blit(font_info.render(f"Ep Reward: {stats['ep_reward']:.1f}", True, (255, 215, 0)), (col1_x, y_start + line_h * 2))
    screen.blit(font_info.render(f"All-Time Best: {stats['best_reward']:.1f}", True, (50, 255, 120)), (col1_x, y_start + line_h * 3))

    screen.blit(font_info.render(f"Forward Vel (vx): {stats['fwd_vel']:+.2f} m/s", True, (0, 220, 255)), (col2_x, y_start))
    screen.blit(font_info.render(f"Torso Height (z): {stats['torso_z']:.2f} m", True, (240, 240, 240)), (col2_x, y_start + line_h))
    screen.blit(font_info.render(f"Critic Loss: {last_losses.get('critic_loss', 0.0):.2f}", True, (240, 240, 240)), (col2_x, y_start + line_h * 2))
    screen.blit(font_info.render(f"Entropy (alpha): {last_losses.get('alpha', 0.0):.4f}", True, (240, 240, 240)), (col2_x, y_start + line_h * 3))


def main():
    # 1. Initialize Pygame Window
    pygame.init()
    pygame.font.init()
    win_w, win_h = 960, 680
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("Humanoid-v5 SAC Live Training Dashboard")
    clock = pygame.time.Clock()

    # 2. Initialize MuJoCo Environment with RescaleAction [-1.0, 1.0]
    raw_env = gym.make("Humanoid-v5", render_mode="rgb_array")
    env = gym.wrappers.RescaleAction(raw_env, min_action=-1.0, max_action=1.0)
    obs, info = env.reset()
    obs_dim = obs.shape[0]
    act_dim = env.action_space.shape[0]

    # 3. Initialize Agent & Replay Buffer
    checkpoint_dir = "checkpoints"
    latest_checkpoint = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
    best_checkpoint = os.path.join(checkpoint_dir, "best_model.pt")

    agent = SACAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        auto_alpha=True,
    )
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, capacity=1_000_000)

    # 4. Check & Resume Checkpoint
    start_step, start_episode, best_reward = agent.load(latest_checkpoint)
    total_steps = start_step
    episode_count = start_episode

    # State variables
    is_fast = True
    is_paused = False
    save_banner_timer = 0
    batch_size = 256
    warmup_steps = 3000 if total_steps == 0 else 0
    save_freq_steps = 25_000

    # Logging Setup
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "training_log.csv")
    if not os.path.exists(log_file):
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "total_steps", "ep_reward", "ep_length", "mean_20", "best_reward", "avg_fwd_vel", "critic_loss", "actor_loss", "alpha"])

    recent_returns = deque(maxlen=20)
    last_losses = {"critic_loss": 0.0, "actor_loss": 0.0, "alpha": agent.alpha}
    ep_fwd_vels = []

    ep_reward = 0.0
    ep_steps = 0
    stats = {
        "step": total_steps,
        "episode": episode_count,
        "ep_reward": 0.0,
        "best_reward": best_reward if best_reward != -float("inf") else 0.0,
        "torso_z": float(obs[0]),
        "fwd_vel": 0.0,
    }

    print("=" * 88)
    print(" HUMANOID-v5 LIVE TRAINING MONITOR INITIALIZED")
    print(" Controls: [SPACE] Pause | [V] Fast/Slow Toggle | [S] Instant Save | [ESC] Exit")
    print(f" Resumed Step: {total_steps:,} | Ep: {episode_count:,} | Best Score: {stats['best_reward']:.1f}")
    print(" Logs auto-saved to: logs/training_log.csv")
    print("=" * 88)

    running = True
    while running:
        # Handle Pygame Events & Hotkeys
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("\n[EXIT] Window closed by user. Saving checkpoint...")
                agent.save(latest_checkpoint, total_steps, episode_count, stats["best_reward"])
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    is_paused = not is_paused
                    print(f"[HOTKEY] Spacebar pressed -> Paused: {is_paused}")
                elif event.key == pygame.K_v:
                    is_fast = not is_fast
                    mode_name = "TURBO (Max Speed)" if is_fast else "LIVE 3D (50 FPS)"
                    print(f"[HOTKEY] 'V' pressed -> Speed Mode: {mode_name}")
                elif event.key == pygame.K_s:
                    agent.save(latest_checkpoint, total_steps, episode_count, stats["best_reward"])
                    save_banner_timer = 90  # display for ~90 frames
                    print(f"[HOTKEY] 'S' pressed -> Instant Checkpoint Saved at step {total_steps:,}")
                elif event.key == pygame.K_ESCAPE:
                    print("\n[EXIT] ESC pressed. Saving checkpoint...")
                    agent.save(latest_checkpoint, total_steps, episode_count, stats["best_reward"])
                    running = False

        if not running:
            break

        # If Paused, redraw HUD and wait
        if is_paused:
            hud_surf = create_hud_surface(win_w, win_h, stats, is_fast, is_paused, save_banner_timer)
            screen.blit(hud_surf, (0, 0))
            pygame.display.flip()
            clock.tick(30)
            continue

        # Environment Step
        if total_steps < warmup_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs, deterministic=False)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Update stats
        ep_reward += float(reward)
        ep_steps += 1
        total_steps += 1

        fwd_v = float(info.get("x_velocity", 0.0))
        ep_fwd_vels.append(fwd_v)
        stats["step"] = total_steps
        stats["episode"] = episode_count
        stats["ep_reward"] = ep_reward
        stats["torso_z"] = float(next_obs[0])
        stats["fwd_vel"] = fwd_v

        # Store in Replay Buffer
        norm_obs = agent.normalizer.normalize(obs)
        norm_next_obs = agent.normalizer.normalize(next_obs)
        replay_buffer.store(norm_obs, action, reward, norm_next_obs, terminated)

        # Policy Update Step
        if total_steps >= warmup_steps and replay_buffer.size >= batch_size:
            batch = replay_buffer.sample(batch_size)
            last_losses = agent.update(batch)

        obs = next_obs

        # Render Frame & Display Handling
        if not is_fast:
            # LIVE 3D Mode: Full 3D rendering at smooth 50 FPS
            frame = env.render()
            if frame is not None:
                surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
                scaled_surf = pygame.transform.scale(surf, (win_w, win_h))
                screen.blit(scaled_surf, (0, 0))

            if save_banner_timer > 0:
                save_banner_timer -= 1
            hud_surf = create_hud_surface(win_w, win_h, stats, is_fast, is_paused, save_banner_timer)
            screen.blit(hud_surf, (0, 0))
            pygame.display.flip()
            clock.tick(50)
        else:
            # TURBO Mode: Bypass 3D rendering for maximum learning throughput (~200+ steps/sec)
            # Update telemetry display periodically so window stays responsive and informative
            if total_steps % 15 == 0 or done:
                screen.fill((12, 16, 24))
                if save_banner_timer > 0:
                    save_banner_timer -= 1
                hud_surf = create_hud_surface(win_w, win_h, stats, is_fast, is_paused, save_banner_timer)
                screen.blit(hud_surf, (0, 0))
                draw_turbo_card(screen, win_w, win_h, stats, last_losses)
                pygame.display.flip()

        # Episode Reset Handling & Terminal Logging
        if done:
            episode_count += 1
            recent_returns.append(ep_reward)
            mean_20 = float(np.mean(recent_returns))
            avg_vel = float(np.mean(ep_fwd_vels)) if len(ep_fwd_vels) > 0 else 0.0

            is_new_best = ep_reward > stats["best_reward"]
            if is_new_best:
                prev_best = stats["best_reward"]
                stats["best_reward"] = ep_reward
                agent.save(best_checkpoint, total_steps, episode_count, stats["best_reward"])

            # Clean Terminal Progress Log
            log_line = (
                f"[EP {episode_count:>5}] "
                f"Steps: {total_steps:>8,} | "
                f"Score: {ep_reward:>7.1f} | "
                f"Len: {ep_steps:>4} | "
                f"Mean(20): {mean_20:>7.1f} | "
                f"Best: {stats['best_reward']:>7.1f} | "
                f"Fwd Vel: {avg_vel:>+5.2f} m/s | "
                f"α: {last_losses['alpha']:.3f} | "
                f"Crit Loss: {last_losses['critic_loss']:>6.3f} | "
                f"Act Loss: {last_losses['actor_loss']:>6.3f}"
            )
            print(log_line)

            if is_new_best:
                print(f"       >>> 🔥 NEW BEST SCORE: {ep_reward:.1f} (Previous: {prev_best:.1f}) | Saved Checkpoint! <<<")

            # Append to CSV for persistent analysis
            try:
                with open(log_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        episode_count,
                        total_steps,
                        round(ep_reward, 2),
                        ep_steps,
                        round(mean_20, 2),
                        round(stats["best_reward"], 2),
                        round(avg_vel, 3),
                        round(last_losses["critic_loss"], 4),
                        round(last_losses["actor_loss"], 4),
                        round(last_losses["alpha"], 4)
                    ])
            except Exception:
                pass

            obs, info = env.reset()
            ep_reward = 0.0
            ep_steps = 0
            ep_fwd_vels = []

        # Periodic Auto-Save
        if total_steps % save_freq_steps == 0:
            agent.save(latest_checkpoint, total_steps, episode_count, stats["best_reward"])

    env.close()
    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
