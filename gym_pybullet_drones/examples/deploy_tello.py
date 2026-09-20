import time
import argparse
import numpy as np
import cv2
import torch
from stable_baselines3 import PPO

try:
    from djitellopy import Tello
except ImportError:
    print("[ERROR] djitellopy not installed. Please install with: pip install djitellopy")
    exit(1)
    
try:
    from transformers import pipeline
except ImportError:
    print("[ERROR] transformers not installed. Please install with: pip install transformers")
    exit(1)

# Import our custom extractor so SB3 can load the model properly
from gym_pybullet_drones.utils.NavigateExtractor import NavigateExtractor

class TelloAgent:
    def __init__(self, model_path, target_pos):
        print(f"[INFO] Loading model from {model_path}...")
        
        # SB3 uses the custom class mapping if we pass custom_objects, 
        # but importing NavigateExtractor is usually enough for it to be found in globals.
        self.model = PPO.load(model_path)
        
        self.tello = Tello()
        self.target = np.array(target_pos, dtype=np.float32)
        
        # Dead reckoning estimates
        self.est_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.last_time = time.time()
        
        print("[INFO] Loading MiDaS depth estimation model (Intel/dps-small)...")
        # Load small depth estimation model
        self.depth_estimator = pipeline("depth-estimation")
        
    def get_observation(self):
        # 1. Image Capture & Depth Estimation
        frame = self.tello.get_frame_read().frame
        if frame is None:
            # Fallback if frame dropped
            depth_map = np.zeros((1, 48, 64), dtype=np.float32)
        else:
            # Resize for MiDaS to speed up processing
            frame_resized = cv2.resize(frame, (256, 192))
            
            # Predict depth
            from PIL import Image
            img_pil = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
            result = self.depth_estimator(img_pil)
            depth_img = np.array(result["depth"])
            
            # Resize to policy input size (64x48)
            depth_small = cv2.resize(depth_img, (64, 48))
            
            # Normalize to 0-1
            depth_min, depth_max = depth_small.min(), depth_small.max()
            if depth_max > depth_min:
                depth_norm = (depth_small - depth_min) / (depth_max - depth_min)
            else:
                depth_norm = depth_small
                
            depth_map = np.expand_dims(depth_norm, axis=0).astype(np.float32)
            
        # 2. State from Tello SDK
        roll = np.radians(self.tello.get_roll())
        pitch = np.radians(self.tello.get_pitch())
        yaw = np.radians(self.tello.get_yaw())
        
        vx = self.tello.get_speed_x() / 100.0 # cm/s -> m/s
        vy = self.tello.get_speed_y() / 100.0
        vz = self.tello.get_speed_z() / 100.0
        
        height = self.tello.get_height() / 100.0 # cm -> m
        
        # 3. Dead reckoning update
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        self.est_pos[0] += vx * dt
        self.est_pos[1] += vy * dt
        
        # 4. Target relative vector
        dx = self.target[0] - self.est_pos[0]
        dy = self.target[1] - self.est_pos[1]
        dz = self.target[2] - height
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        heading = np.arctan2(dy, dx) - yaw
        
        state = np.array([
            roll, pitch, yaw,
            vx, vy, vz,
            height,
            self.est_pos[0], self.est_pos[1],
            dx, dy, dz, dist, heading, 0.0
        ], dtype=np.float32)
        
        return {"depth": depth_map, "state": state}
        
    def run(self):
        print("[INFO] Connecting to Tello...")
        self.tello.connect()
        print(f"[INFO] Battery level: {self.tello.get_battery()}%")
        
        if self.tello.get_battery() < 20:
            print("[ERROR] Battery too low! (<20%)")
            return
            
        self.tello.streamon()
        print("[INFO] Taking off...")
        self.tello.takeoff()
        
        # Move up a bit to reach initial height ~1.0m
        self.tello.move_up(30)
        
        self.last_time = time.time()
        
        try:
            while True:
                obs = self.get_observation()
                
                # Predict action
                # We need to add batch dimension for SB3
                obs_batch = {
                    "depth": np.expand_dims(obs["depth"], axis=0),
                    "state": np.expand_dims(obs["state"], axis=0)
                }
                
                action, _ = self.model.predict(obs_batch, deterministic=True)
                action = action[0] # Remove batch dim
                
                # Action mapping: [-1, 1] to [-100, 100]
                lr = int(np.clip(action[0] * 100, -100, 100))
                fb = int(np.clip(action[1] * 100, -100, 100))
                ud = int(np.clip(action[2] * 100, -100, 100))
                yaw_vel = int(np.clip(action[3] * 100, -100, 100))
                
                print(f"Dist: {obs['state'][12]:.2f}m | Action: lr={lr}, fb={fb}, ud={ud}, yaw={yaw_vel}")
                
                self.tello.send_rc_control(lr, fb, ud, yaw_vel)
                
                # Check arrival
                if obs["state"][12] < 0.15: # 15cm threshold
                    print("[INFO] TARGET REACHED!")
                    self.tello.send_rc_control(0, 0, 0, 0)
                    break
                    
                # Safety checks
                if obs["state"][6] > 2.5: # Too high
                    print("[WARNING] Max altitude exceeded! Aborting.")
                    break
                if abs(obs["state"][0]) > 0.7 or abs(obs["state"][1]) > 0.7:
                    print("[WARNING] Excessive tilt detected! Aborting.")
                    break
                    
                time.sleep(0.05) # ~20Hz control loop
                
        except KeyboardInterrupt:
            print("\n[INFO] Manual interrupt detected!")
        except Exception as e:
            print(f"\n[ERROR] Exception occurred: {e}")
        finally:
            print("[INFO] Landing...")
            self.tello.send_rc_control(0, 0, 0, 0)
            self.tello.land()
            self.tello.streamoff()
            print("[INFO] Disconnected.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deploy NavigateAviary on real Ryze Tello')
    parser.add_argument('--model', type=str, required=True, help='Path to best_model.zip')
    parser.add_argument('--target', type=float, nargs=3, default=[2.0, 0.0, 1.0], help='Target X Y Z relative to takeoff point (meters)')
    
    args = parser.parse_args()
    
    agent = TelloAgent(model_path=args.model, target_pos=args.target)
    agent.run()
