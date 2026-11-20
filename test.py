import os
import sys
import argparse
import pygame
import numpy as np
import gymnasium as gym
from agent import SACAgent


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Humanoid-v5 Model Evaluation (Visual 3D & Statistics)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python test.py                     # Evaluate 'best_model.pt' continuously
  python test.py latest              # Evaluate 'latest_checkpoint.pt' continuously
  python test.py best 5              # Evaluate 'best' model for 5 episodes
  python test.py latest 10           # Evaluate 'latest' model for 10 episodes
  python test.py --model best -n 20  # Evaluate 'best' model for 20 episodes
"""
    )
    parser.add_argument("target", nargs="?", default=None, help="Model target (best/latest) or episode count")
    parser.add_argument("count", nargs="?", type=int, default=None, help="Number of episodes to evaluate")
    parser.add_argument("--model", "-m", type=str, default=None, help="Model to evaluate ('best', 'latest', or path)")
    parser.add_argument("--episodes", "-n", type=int, default=None, help="Number of episodes (default: continuous)")
    args = parser.parse_args()

    # Flexible position/flag resolution
    model_arg = args.model or args.target
    ep_count = args.episodes or args.count

    if model_arg and model_arg.isdigit() and ep_count is None:
        ep_count = int(model_arg)
        model_arg = None

    if not model_arg:
        model_arg = "best"

    # Resolve Checkpoint Path
    model_lower = model_arg.lower().strip()
    checkpoint_dir = "checkpoints"

    if model_lower in ["best", "best_model", "record"]:
        ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
        model_tag = "BEST RECORD MODEL"
    elif model_lower in ["latest", "last", "current", "checkpoint"]:
        ckpt_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
        model_tag = "LATEST CHECKPOINT"
    elif model_lower in ["backup", "1m", "backup_1m"]:
        ckpt_path = os.path.join(checkpoint_dir, "backup_1m_milestone", "latest_checkpoint.pt")
        model_tag = "1M MILESTONE BACKUP"
    else:
        ckpt_path = model_arg
        model_tag = os.path.basename(model_arg)

    if not os.path.exists(ckpt_path):
        print(f"\n[ERROR] Checkpoint not found at: {ckpt_path}")
        print("Available options: 'best', 'latest', or a direct file path.")
        sys.exit(1)

    return ckpt_path, model_tag, ep_count


def main():
    checkpoint_path, model_tag, max_episodes = parse_arguments()

    print("=" * 72)
    print(" HUMANOID-v5 MODEL EVALUATION (DETERMINISTIC SPRINT TEST)")
    print(f" Target Model:  {model_tag} ({checkpoint_path})")
    ep_target_str = f"{max_episodes} Episodes" if max_episodes else "Continuous (Exit anytime with [ESC])"
    print(f" Evaluation:    {ep_target_str}")
    print(" Camera:        Drone Chase Tracking Camera (Centred on Torso)")
    print(" Controls:      [SPACE] Pause | [ESC] Exit & Show Summary")
    print("=" * 72)

    # 1. Setup Pygame Window
    pygame.init()
    pygame.font.init()
    win_w, win_h = 960, 680
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption(f"Humanoid-v5 Evaluation | {model_tag}")
    clock = pygame.time.Clock()

    font_large = pygame.font.SysFont("Consolas", 16, bold=True)
    font_med = pygame.font.SysFont("Consolas", 13)

    # 2. Initialize MuJoCo Environment with RescaleAction & Drone Tracking Camera
    raw_env = gym.make("Humanoid-v5", render_mode="rgb_array", camera_name="track")
    env = gym.wrappers.RescaleAction(raw_env, min_action=-1.0, max_action=1.0)
    obs, info = env.reset()
    obs_dim = obs.shape[0]
    act_dim = env.action_space.shape[0]

    # 3. Load Agent
    agent = SACAgent(obs_dim=obs_dim, act_dim=act_dim)
    step, episode, best_reward = agent.load(checkpoint_path)

    test_episodes = 0
    ep_reward = 0.0
    ep_steps = 0
    ep_vels = []
    all_scores = []
    all_lens = []
    all_avg_vels = []
    is_paused = False

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                is_paused = not is_paused

        if not running:
            break

        if is_paused:
            clock.tick(30)
            continue

        # Deterministic Action Selection (Zero Motor Noise)
        action = agent.select_action(obs, deterministic=True, update_normalizer=False)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        ep_reward += float(reward)
        ep_steps += 1
        cur_vel = float(info.get("x_velocity", 0.0))
        ep_vels.append(cur_vel)
        obs = next_obs

        # Render 3D Frame with Tracking Camera
        frame = env.render()
        if frame is not None:
            surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            scaled_surf = pygame.transform.scale(surf, (win_w, win_h))
            screen.blit(scaled_surf, (0, 0))

        # HUD Overlay
        header_h = 70
        hud = pygame.Surface((win_w, header_h), pygame.SRCALPHA)
        pygame.draw.rect(hud, (15, 20, 30, 230), (0, 0, win_w, header_h))
        pygame.draw.line(hud, (0, 220, 150, 200), (0, header_h), (win_w, header_h), 2)

        # Title & Mode
        title_str = f"HUMANOID-v5 | {model_tag} (Trained: {step:,} steps)"
        title = font_large.render(title_str, True, (0, 255, 180))
        hud.blit(title, (15, 10))

        pause_tag = "|| PAUSED" if is_paused else ">> LIVE 3D (50 FPS)"
        pause_surf = font_med.render(pause_tag, True, (255, 200, 0) if is_paused else (0, 255, 120))
        hud.blit(pause_surf, (win_w - pause_surf.get_width() - 15, 10))

        # Stats Row
        avg_score = np.mean(all_scores) if len(all_scores) > 0 else 0.0
        ep_display_str = f"Ep: {test_episodes + 1}/{max_episodes}" if max_episodes else f"Ep: {test_episodes + 1}"
        row2 = f"{ep_display_str}  |  Score: {ep_reward:7.1f}  |  Avg Test Score: {avg_score:7.1f}  |  Record: {best_reward:.1f}"
        hud.blit(font_med.render(row2, True, (255, 230, 100)), (15, 38))

        kinematics = f"Height: {float(obs[0]):.2f} m  |  Fwd Vel: {cur_vel:+.2f} m/s"
        kin_surf = font_med.render(kinematics, True, (150, 220, 255))
        hud.blit(kin_surf, (win_w - kin_surf.get_width() - 15, 38))

        screen.blit(hud, (0, 0))
        pygame.display.flip()
        clock.tick(50)  # Smooth 50 FPS replay

        if done:
            test_episodes += 1
            all_scores.append(ep_reward)
            all_lens.append(ep_steps)
            mean_v = float(np.mean(ep_vels)) if ep_vels else 0.0
            all_avg_vels.append(mean_v)

            target_tag = f"/{max_episodes}" if max_episodes else ""
            print(f"[TEST EP {test_episodes:>3}{target_tag}] Score: {ep_reward:>7.1f} | Length: {ep_steps:>4} steps | "
                  f"Avg Fwd Vel: {mean_v:+.2f} m/s | Running Mean: {np.mean(all_scores):>7.1f}")

            # Check if requested episodes reached
            if max_episodes and test_episodes >= max_episodes:
                running = False
                break

            obs, info = env.reset()
            ep_reward = 0.0
            ep_steps = 0
            ep_vels = []

    env.close()
    pygame.quit()

    # Final Summary Report
    print("\n" + "=" * 65)
    print("                     EVALUATION SUMMARY REPORT                   ")
    print("=" * 65)
    print(f" Model Evaluated:       {model_tag}")
    print(f" Checkpoint Path:       {checkpoint_path}")
    print(f" Total Episodes Tested: {len(all_scores)}")

    if len(all_scores) > 0:
        completed_1000 = [s for s, l in zip(all_scores, all_lens) if l >= 1000]
        success_rate = (len(completed_1000) / len(all_scores)) * 100.0
        print(f" Average Score:         {np.mean(all_scores):.1f}")
        print(f" Max Score:             {np.max(all_scores):.1f}")
        print(f" Min Score:             {np.min(all_scores):.1f}")
        print(f" Full 1,000-Step Runs:  {len(completed_1000)} / {len(all_scores)} ({success_rate:.1f}% Success Rate)")
        if completed_1000:
            print(f" Avg Sprint Score:      {np.mean(completed_1000):.1f} (on completed runs)")
        print(f" Overall Avg Velocity:  {np.mean(all_avg_vels):+.2f} m/s")
    print("=" * 65 + "\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
