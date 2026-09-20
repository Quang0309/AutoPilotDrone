import os
import time
import argparse
import numpy as np
import pybullet as p
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

from gym_pybullet_drones.envs.NavigateAviary import NavigateAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.utils import sync, str2bool

DEFAULT_MODEL_PATH = 'results/navigate/best_model.zip'
DEFAULT_GUI = True
DEFAULT_RECORD = True
DEFAULT_EPISODES = 5

def run(model_path=DEFAULT_MODEL_PATH, gui=DEFAULT_GUI, episodes=DEFAULT_EPISODES, record_video=DEFAULT_RECORD):
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found at {model_path}")
        return
        
    print(f"[INFO] Loading model from {model_path}")
    model = PPO.load(model_path)
    
    base_env = NavigateAviary(gui=gui, obs=ObservationType.ALL, act=ActionType.VEL, record=record_video)
    test_env = DummyVecEnv([lambda: base_env])
    test_env = VecFrameStack(test_env, n_stack=4, channels_order="first")
    
    successes = 0
    collisions = 0
    
    for ep in range(episodes):
        obs = test_env.reset()
        start = time.time()
        print(f"\n--- Episode {ep+1} ---")
        print(f"Target position: {base_env.TARGET_POS}")
        
        for i in range(int((base_env.EPISODE_LEN_SEC) * base_env.CTRL_FREQ)):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)
            
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
                    print(f"Episode {ep+1} Finished: SUCCESS! Reached target.")
                    successes += 1
                elif inf.get("collision", False):
                    print(f"Episode {ep+1} Finished: COLLISION detected!")
                    collisions += 1
                else:
                    print(f"Episode {ep+1} Finished: Truncated (timeout or out of bounds).")
                break
                
    test_env.close()
    
    print("\n--- Testing Summary ---")
    print(f"Total Episodes: {episodes}")
    print(f"Successes: {successes} ({(successes/episodes)*100:.1f}%)")
    print(f"Collisions: {collisions} ({(collisions/episodes)*100:.1f}%)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test trained NavigateAviary model')
    parser.add_argument('--model_path', default=DEFAULT_MODEL_PATH, type=str, help='Path to best_model.zip')
    parser.add_argument('--gui', default=DEFAULT_GUI, type=str2bool, help='Whether to use PyBullet GUI (default: True)')
    parser.add_argument('--episodes', default=DEFAULT_EPISODES, type=int, help='Number of episodes to test (default: 5)')
    parser.add_argument('--record_video', default=DEFAULT_RECORD, type=str2bool, help='Whether to record video frames (default: True)')
    
    ARGS = parser.parse_args()
    run(**vars(ARGS))
