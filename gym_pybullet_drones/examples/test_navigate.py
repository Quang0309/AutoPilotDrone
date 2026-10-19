import os
import time
import argparse
import cv2
import numpy as np
import pybullet as p
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

from gym_pybullet_drones.envs.NavigateAviary import NavigateAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.utils import sync, str2bool

DEFAULT_MODEL_PATH = 'results/navigate/best_model.zip'
DEFAULT_GUI = True
DEFAULT_RECORD = False
DEFAULT_EPISODES = 5
DEFAULT_SHOW_CAMERA = True

def create_hud(base_env, action, step=1, ep=1, episodes=5, status="NAVIGATING"):
    """
    Renders a live Heads-Up Display (HUD) showing:
      1. Left: Drone First-Person View (RGB Camera) with crosshair
      2. Right: CNN Depth Perception (Colored Depth Map)
      3. Bottom: Flight Telemetry & Control Actions
    """
    rgb, dep, _ = base_env._getDroneImages(0, segmentation=False)
    
    # 1. First-Person RGB View
    rgb_bgr = cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2BGR)
    rgb_bgr = cv2.resize(rgb_bgr, (320, 240), interpolation=cv2.INTER_NEAREST)
    # Center crosshair to show where the drone nose is pointing
    cv2.drawMarker(rgb_bgr, (160, 120), (0, 255, 0), cv2.MARKER_CROSS, 16, 1)
    cv2.putText(rgb_bgr, "DRONE FPV (CAMERA)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    
    # 2. CNN Depth Perception (Colormapped depth)
    near = base_env.L
    far = 1000.0
    dep_clamped = np.clip(np.nan_to_num(dep, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    denom = np.maximum(far - (far - near) * dep_clamped, 1e-4)
    depth_linear = far * near / denom
    dep_norm = np.clip(depth_linear, 0.0, 5.0) / 5.0
    depth_vis = (dep_norm * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
    depth_colored = cv2.resize(depth_colored, (320, 240), interpolation=cv2.INTER_NEAREST)
    cv2.putText(depth_colored, "CNN DEPTH PERCEPTION", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    video_row = np.hstack([rgb_bgr, depth_colored])
    
    # 3. Telemetry Bar
    bar = np.full((75, 640, 3), 25, dtype=np.uint8)
    
    state = base_env._getDroneStateVector(0)
    pos = state[0:3]
    dist = np.linalg.norm(base_env.TARGET_POS - pos)
    speed = np.linalg.norm(state[10:13])
    yaw = state[9]
    dx = base_env.TARGET_POS[0] - pos[0]
    dy = base_env.TARGET_POS[1] - pos[1]
    heading = (np.arctan2(dy, dx) - yaw + np.pi) % (2 * np.pi) - np.pi
    heading_deg = np.degrees(heading)
    
    status_color = (0, 255, 0) if status == "SUCCESS" else ((0, 0, 255) if status == "COLLISION" else (0, 255, 255))
    
    cv2.putText(bar, f"EPISODE: {ep}/{episodes} | STEP: {step:03d} | STATUS: {status}", (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1, cv2.LINE_AA)
    cv2.putText(bar, f"Dist to Goal: {dist:.2f}m | Speed: {speed:.2f}m/s | Heading: {heading_deg:+.1f} deg", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(bar, f"Cmd: Roll={action[0]:+.2f}  Pitch={action[1]:+.2f}  Thrust={action[2]:+.2f}  Yaw={action[3]:+.2f}", (15, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
    
    frame = np.vstack([video_row, bar])
    return frame

def run(model_path=DEFAULT_MODEL_PATH, gui=DEFAULT_GUI, episodes=DEFAULT_EPISODES, record_video=DEFAULT_RECORD, difficulty='easy', show_camera=DEFAULT_SHOW_CAMERA):
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found at {model_path}")
        return
        
    print(f"[INFO] Loading model from {model_path} with difficulty: {difficulty}")
    model = PPO.load(model_path)
    
    # Initialize Environment
    base_env = NavigateAviary(obs=ObservationType.ALL, act=ActionType.VEL, gui=gui, record=record_video, difficulty=difficulty)
    test_env = DummyVecEnv([lambda: base_env])
    test_env = VecFrameStack(test_env, n_stack=4, channels_order="first")
    
    successes = 0
    collisions = 0
    out_of_bounds = 0
    timeouts = 0
    
    for ep in range(episodes):
        obs = test_env.reset()
        start = time.time()
        print(f"\n--- Episode {ep+1} ---")
        print(f"Target position: {base_env.TARGET_POS}")
        
        # Cache target pos for this episode (before auto-reset overwrites it)
        ep_target = base_env.TARGET_POS.copy()
        
        for i in range(int((base_env.EPISODE_LEN_SEC) * base_env.CTRL_FREQ)):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)
            
            # Determine status from the info dict
            # Note: when done=True, DummyVecEnv auto-resets the env, so
            # base_env state is already the NEW episode. Use info dict instead.
            status = "NAVIGATING"
            if done[0]:
                inf = info[0]
                if inf.get("success", False):
                    status = "SUCCESS"
                else:
                    reason = inf.get("truncation_reason", "unknown")
                    status_map = {
                        "collision": "COLLISION",
                        "out_of_bounds": "OUT OF BOUNDS",
                        "flipped": "FLIPPED",
                        "timeout": "TIMEOUT",
                    }
                    status = status_map.get(reason, "TRUNCATED")
                        
            # Show live camera HUD
            if show_camera:
                hud = create_hud(base_env, action[0], step=i+1, ep=ep+1, episodes=episodes, status=status)
                cv2.imshow("Tello Drone Live Vision & Depth Perception", hud)
                wait_time = 1 if not done[0] else 1000
                cv2.waitKey(wait_time)
            
            if gui:
                # 3rd Person Camera Follow
                drone_pos = base_env.pos[0]
                drone_yaw = base_env.rpy[0, 2] # yaw in radians
                
                # PyBullet camera math: subtract 90 degrees to align camera view with drone's forward +X axis
                cam_yaw = np.degrees(drone_yaw) - 90 
                
                p.resetDebugVisualizerCamera(
                    cameraDistance=2.0,
                    cameraYaw=cam_yaw,
                    cameraPitch=-25,
                    cameraTargetPosition=drone_pos,
                    physicsClientId=base_env.CLIENT
                )
                
                base_env.render()
                sync(i, start, base_env.CTRL_TIMESTEP)
                
            if done[0]:
                inf = info[0]
                if inf.get("success", False):
                    print(f"Episode {ep+1} Finished: SUCCESS! Reached target. (dist={inf.get('distance', '?'):.2f}m)")
                    successes += 1
                else:
                    reason = inf.get("truncation_reason", "unknown")
                    if reason == "collision":
                        print(f"Episode {ep+1} Finished: COLLISION detected!")
                        collisions += 1
                    elif reason == "out_of_bounds":
                        print(f"Episode {ep+1} Finished: OUT OF BOUNDS (left the cage).")
                        out_of_bounds += 1
                    elif reason == "flipped":
                        print(f"Episode {ep+1} Finished: FLIPPED (excessive tilt).")
                        out_of_bounds += 1
                    elif reason == "timeout":
                        print(f"Episode {ep+1} Finished: TIMEOUT (max time {base_env.EPISODE_LEN_SEC}s reached).")
                        timeouts += 1
                    else:
                        print(f"Episode {ep+1} Finished: TRUNCATED (reason: {reason}).")
                break
        else:
            print(f"Episode {ep+1} Finished: TIMEOUT! Reached maximum step limit.")
            timeouts += 1

    test_env.close()
    if show_camera:
        cv2.destroyAllWindows()
    
    print("\n--- Testing Summary ---")
    print(f"Total Episodes: {episodes}")
    print(f"Successes:      {successes} ({(successes/episodes)*100:.1f}%)")
    print(f"Collisions:     {collisions} ({(collisions/episodes)*100:.1f}%)")
    print(f"Out of Bounds:  {out_of_bounds} ({(out_of_bounds/episodes)*100:.1f}%)")
    print(f"Timeouts:       {timeouts} ({(timeouts/episodes)*100:.1f}%)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test trained NavigateAviary model')
    parser.add_argument('--model_path', default=DEFAULT_MODEL_PATH, type=str, help='Path to best_model.zip')
    parser.add_argument('--gui', default=DEFAULT_GUI, type=str2bool, help='Whether to use PyBullet GUI (default: True)')
    parser.add_argument('--episodes', default=DEFAULT_EPISODES, type=int, help='Number of episodes to test (default: 5)')
    parser.add_argument('--record_video', default=DEFAULT_RECORD, type=str2bool, help='Whether to record video frames (default: False)')
    parser.add_argument('--difficulty', default='easy', type=str, choices=['easy', 'medium', 'hard'], help='Difficulty used during training (easy, medium, or hard)')
    parser.add_argument('--show_camera', default=DEFAULT_SHOW_CAMERA, type=str2bool, help='Whether to show live First-Person camera and CNN depth view (default: True)')
    
    ARGS = parser.parse_args()
    run(**vars(ARGS))

