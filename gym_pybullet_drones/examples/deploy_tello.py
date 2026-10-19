"""
Real-World Deployment Script for Project ICARUS Autonomous Drone Navigation
Deploys trained SB3 PPO policy (from results/navigate/stage_3_icarus/best_model.zip)
to a physical DJI Tello / Ryze Tello micro-quadrotor over WiFi UDP.

Features:
- Asynchronous threaded MiDaS-Small depth estimation (runs on Apple Silicon GPU / MPS)
- 4-frame temporal stacking matching VecFrameStack(n_stack=4)
- Exact 15-dimensional body-frame kinematic state alignment
- Proportional 0.6m obstacle forward speed governor
- Autopilot Auto-Yaw and Altitude Hold matching Stage 3 ICARUS architecture
- Live OpenCV FPV + Turbo Depth HUD with Spacebar emergency kill-switch
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import time
import threading
import argparse
from collections import deque
import numpy as np
import cv2
import torch
from stable_baselines3 import PPO

try:
    from djitellopy import Tello
except ImportError:
    print("[ERROR] djitellopy not installed. Please run: pip install djitellopy")
    exit(1)

# Import custom feature extractor so SB3 can deserialize the policy
from gym_pybullet_drones.utils.NavigateExtractor import NavigateExtractor


class DepthEstimatorWorker(threading.Thread):
    """
    Asynchronous vision worker that continuously reads the Tello RGB camera,
    infers metric-like normalized depth using MiDaS-Small on MPS/CPU,
    and publishes the latest 48x64 depth map without blocking the 20 Hz flight loop.
    """
    def __init__(self, tello_frame_reader):
        super().__init__(daemon=True)
        self.frame_reader = tello_frame_reader
        self.running = True
        self.lock = threading.Lock()
        
        # Output depth map (48, 64) normalized to [0, 1] (0 = contact, 1 = open air)
        self.latest_depth_map = np.ones((48, 64), dtype=np.float32)
        self.latest_vis_frame = None
        self.min_forward_dist_m = 5.0
        
        print("[VISION] Initializing MiDaS-Small depth estimator...")
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small').to(self.device).eval()
        self.midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms').small_transform
        print(f"[VISION] MiDaS-Small loaded on {self.device}.")

    def run(self):
        while self.running:
            frame = self.frame_reader.frame
            if frame is None:
                time.sleep(0.02)
                continue
            
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_batch = self.midas_transforms(img_rgb).to(self.device)
            
            with torch.no_grad():
                prediction = self.midas(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=(48, 64),
                    mode='bicubic',
                    align_corners=False
                ).squeeze()
                depth_raw = prediction.cpu().numpy()
                
            # MiDaS outputs inverse depth (disparity). Invert & normalize so 0 = near, 1 = far
            disp_min, disp_max = np.percentile(depth_raw, 2), np.percentile(depth_raw, 98)
            if disp_max > disp_min:
                # High disparity = close; invert so close = 0.0, far = 1.0 (matching PyBullet)
                depth_norm = np.clip((disp_max - depth_raw) / (disp_max - disp_min + 1e-6), 0.0, 1.0)
            else:
                depth_norm = np.full_like(depth_raw, 0.5)
                
            # Calculate forward obstacle clearance from central 40% region
            center_roi = depth_norm[int(48 * 0.3):int(48 * 0.7), int(64 * 0.3):int(64 * 0.7)]
            closest_roi = np.percentile(center_roi, 5) # 5th percentile ignores speckle noise
            est_dist_m = float(closest_roi * 3.5)      # Scaled roughly to indoor meters
            
            # Prepare Turbo colormap for HUD visualization
            depth_vis = (depth_norm * 255).astype(np.uint8)
            depth_vis_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
            
            with self.lock:
                self.latest_depth_map = depth_norm.astype(np.float32) # (48, 64)
                self.latest_vis_frame = depth_vis_color
                self.min_forward_dist_m = est_dist_m
                
            time.sleep(0.01)

    def get_latest(self):
        with self.lock:
            return self.latest_depth_map.copy(), self.latest_vis_frame, self.min_forward_dist_m

    def stop(self):
        self.running = False


class RealTelloIcarusAgent:
    def __init__(self, model_path, target_pos, max_speed_cm=35, vps_unit="dm", vps_x_sign=-1, vps_y_sign=1, policy_yaw=False, use_pad=False, pad_at="target", pad_pos=None):
        print("=" * 60)
        print("  PROJECT ICARUS: REAL-WORLD DRONE DEPLOYMENT INTERFACE")
        print("=" * 60)
        print(f"[INIT] Target Waypoint (X, Y, Z): {target_pos} meters")
        print(f"[INIT] Max Forward Cruising Speed: {max_speed_cm} cm/s")
        print(f"[INIT] VPS Telemetry Unit       : {vps_unit.upper()}/s (scale: {'/10.0' if vps_unit == 'dm' else '/100.0'})")
        self.vps_x_sign = float(vps_x_sign)
        self.vps_y_sign = float(vps_y_sign)
        print(f"[INIT] VPS X-Axis Direction     : {'Inverted (-vgx)' if self.vps_x_sign < 0 else 'Direct (+vgx)'}")
        print(f"[INIT] VPS Y-Axis Direction     : {'Direct (+vgy)' if self.vps_y_sign > 0 else 'Inverted (-vgy)'}")
        self.policy_yaw = policy_yaw
        print(f"[INIT] Yaw Controller           : {'Neural Policy (action[3])' if self.policy_yaw else 'Autopilot Waypoint Tracking'}")
        self.use_pad = use_pad
        self.pad_at = pad_at
        self.pad_pos = pad_pos
        self.last_pad_info = (-1, 0.0, 0.0, 0.0) # (mid, x_m, y_m, z_m)
        if self.use_pad:
            loc_str = f"Custom {pad_pos}" if pad_pos else f"{pad_at.upper()} ({target_pos[0]:.2f}, {target_pos[1]:.2f})m" if pad_at == "target" else "ORIGIN (0.0, 0.0)m"
            print(f"[INIT] DJI Mission Pad Tracking : ENABLED (Location: {loc_str})")
        else:
            print("[INIT] DJI Mission Pad Tracking : DISABLED (Pure VPS optical flow)")
        print(f"[INIT] Loading RL Policy from: {model_path}...")
        
        self.model = PPO.load(model_path)
        print("[INIT] Policy loaded successfully.")
        
        self.target = np.array(target_pos, dtype=np.float32)
        self.max_speed_cm = max_speed_cm
        self.vps_unit = vps_unit
        self.last_vps = [0.0, 0.0, 0.0] # [raw_vgx, raw_vgy, raw_vgz]
        
        # Dead-reckoned 2D position estimates (meters)
        self.est_pos = np.zeros(2, dtype=np.float32)
        self.last_time = time.time()
        
        self.takeoff_yaw_deg = 0.0
        self.last_rc_cmds = [0, 0, 0, 0] # [lr, fb, ud, yaw]
        self.last_raw_tof = 0.0
        self.last_raw_h = 0.0
        
        # 4-frame temporal history buffers for SB3 VecFrameStack
        self.depth_buffer = deque(maxlen=4)
        self.state_buffer = deque(maxlen=4)
        
        # Connect to drone (handling port 8889 conflict if Jupyter notebook held socket)
        try:
            self.tello = Tello()
        except OSError as e:
            if "Address already in use" in str(e) or getattr(e, 'errno', 0) == 48:
                print("\n" + "!" * 70)
                print("[ERROR] Port 8889 is already bound by another Python process or Jupyter kernel!")
                print("[FIX] Run this terminal command to free the port immediately:")
                print("      lsof -ti:8889 | xargs kill -9")
                print("!" * 70 + "\n")
            raise e
        self.depth_worker = None

    @staticmethod
    def map_actuator(action_val, max_cmd, deadband=6, deadzone=0.03):
        """
        Actuator Transfer Function:
        Maps continuous policy action [-1, 1] to physical RC command [-max_cmd, max_cmd].
        Compensates for physical motor static friction (deadband) so small policy commands
        produce real aircraft acceleration rather than being swallowed by the ESC.
        If |action_val| < deadzone, returns 0 (true neutral hover/idle).
        """
        val = float(action_val)
        abs_val = abs(val)
        if abs_val < deadzone:
            return 0
        mag = deadband + (abs_val - deadzone) / (1.0 - deadzone) * (max_cmd - deadband)
        return int(np.clip(np.sign(val) * mag, -max_cmd, max_cmd))

    def connect(self):
        print("\n[TELLO] Connecting to aircraft via WiFi...")
        self.tello.connect()
        battery = self.tello.get_battery()
        temp = self.tello.get_highest_temperature()
        print(f"[TELLO] Connection Established!")
        print(f"[TELLO] Battery Level : {battery}%")
        print(f"[TELLO] Temperature   : {temp}°C")
        
        if battery < 20:
            raise RuntimeError(f"Battery too low ({battery}%)! Please charge before flight.")
            
        self.tello.streamon()
        frame_reader = self.tello.get_frame_read()
        
        # Wait for camera stream to warm up
        print("[TELLO] Initializing camera stream...")
        for _ in range(30):
            if frame_reader.frame is not None:
                break
            time.sleep(0.1)
            
        if frame_reader.frame is None:
            raise RuntimeError("Failed to receive video frames from Tello camera!")
            
        print("[TELLO] Video stream active.")
        
        # Enable DJI Mission Pad detection if requested
        if self.use_pad:
            print("[TELLO] Enabling DJI Mission Pad detection (downward camera @ 20Hz)...")
            try:
                self.tello.enable_mission_pads()
                time.sleep(0.2)
                self.tello.set_mission_pad_detection_direction(0) # 0 = downward camera only
                time.sleep(0.3)
                print("[TELLO] Mission Pad detection active!")
            except Exception as e:
                print(f"[WARN] Failed to enable mission pads: {e}")
        self.depth_worker = DepthEstimatorWorker(frame_reader)
        self.depth_worker.start()
        
        # Allow depth worker to warm up first prediction
        print("[TELLO] Warming up neural depth estimator...")
        time.sleep(1.0)

    def get_kinematics_15d(self):
        """
        Constructs the exact 15-dimensional kinematics vector expected by NavigateAviary:
        [roll, pitch, yaw, vx_body, vy_body, vz, height, est_x, est_y, dx_body, dy_body, dz, dist, heading, 0.0]
        """
        # Relative Yaw Calibration:
        # Tello IMU reports yaw in degrees [-180, 180] where Clockwise (CW) is positive.
        # We zero it at takeoff (takeoff_yaw_deg) so 0 deg always aligns with the cage corridor.
        raw_yaw = float(self.tello.get_yaw())
        rel_yaw_deg = (raw_yaw - self.takeoff_yaw_deg + 180.0) % 360.0 - 180.0
        
        # In PyBullet / Cartesian right-hand rule, Counter-Clockwise (CCW/Left) is positive.
        # Therefore: cartesian_yaw = -rel_yaw_deg.
        yaw = np.radians(-rel_yaw_deg)
        
        # Roll and Pitch (radians)
        roll = np.radians(self.tello.get_roll())
        pitch = np.radians(self.tello.get_pitch())
        
        # Velocities directly from Onboard VPS (Visual Positioning System):
        # In DJI Tello / RoboMaster TT firmware, raw_vgx and raw_vgy are estimated
        # in the horizontal Ground Frame established at BOOT (power-on heading = 0 deg).
        # RoboMaster TT (SDK 3.0) reports velocities in integer dm/s (1 dm/s = 0.1 m/s = 10 cm/s).
        # Standard Tello (SDK 1.3/2.0) reports velocities in cm/s (1 cm/s = 0.01 m/s).
        raw_vgx = float(self.tello.get_speed_x())
        raw_vgy = float(self.tello.get_speed_y())
        raw_vgz = float(self.tello.get_speed_z())
        self.last_vps = [raw_vgx, raw_vgy, raw_vgz]
        
        scale = 10.0 if self.vps_unit == 'dm' else 100.0
        
        # 1. Scaled velocities in Tello Boot Frame:
        # Multiplied by vps_x_sign (-1 by default) to convert into standard Cartesian (+vx = forward).
        vx_boot = (self.vps_x_sign * raw_vgx) / scale
        vy_boot = (self.vps_y_sign * raw_vgy) / scale  # raw_vgy < 0 is rightward (Cartesian -Y)
        vz = raw_vgz / scale
        
        # Sub-decimeter optical flow quantization fallback:
        # Tello SDK 3.0 sends integer dm/s (1 dm/s = 10 cm/s). Any physical speed < 10 cm/s
        # truncates to raw = 0. If the drone is actively commanding stick, estimate true
        # sub-decimeter ground creep so dead-reckoning never stalls:
        if raw_vgx == 0.0 and abs(self.last_rc_cmds[1]) >= 6:
            creep_vx = float(self.last_rc_cmds[1]) * 0.007
            vx_boot += creep_vx
        if raw_vgy == 0.0 and abs(self.last_rc_cmds[0]) >= 4:
            # lr_cmd > 0 is rightward (Cartesian -Y)
            creep_vy = -float(self.last_rc_cmds[0]) * 0.007
            vy_boot += creep_vy
        
        # 2. Rotate from Boot Frame into Cage Corridor Frame (World Frame):
        # The corridor heading relative to the boot frame is takeoff_yaw_deg (in Tello CW deg).
        # In Cartesian CCW radians: corridor_yaw_rad = -takeoff_yaw_deg.
        corridor_yaw_rad = np.radians(-self.takeoff_yaw_deg)
        cos_c = np.cos(corridor_yaw_rad)
        sin_c = np.sin(corridor_yaw_rad)
        
        vx_corridor =  vx_boot * cos_c + vy_boot * sin_c
        vy_corridor = -vx_boot * sin_c + vy_boot * cos_c
        
        # 3. Rotate from Corridor Frame into Drone Body Frame (for 15D state vector):
        # The drone's heading relative to the corridor is yaw = -rel_yaw_deg.
        cos_y = np.cos(yaw)
        sin_y = np.sin(yaw)
        vx_body =  vx_corridor * cos_y + vy_corridor * sin_y
        vy_body = -vx_corridor * sin_y + vy_corridor * cos_y
        
        # Height from downward distance sensors (meters):
        # 1. Primary sensor: get_distance_tof() (Downward infrared Time-of-Flight laser)
        #    Reports direct physical ground clearance in centimeters (cm).
        # 2. Secondary fallback: get_height() (Fused barometer/takeoff relative altitude)
        #    NOTE: In Tello SDK 3.0 / RoboMaster TT, 'h' is reported in decimeters (dm)!
        #    e.g. 10 dm = 100 cm = 1.0 m. Dividing 10 by 100 gave 0.10m by mistake.
        raw_tof = float(self.tello.get_distance_tof())
        raw_h = float(self.tello.get_height())
        self.last_raw_tof = raw_tof
        self.last_raw_h = raw_h
        
        if 10.0 <= raw_tof <= 300.0:
            height = raw_tof / 100.0
        else:
            # Fallback to get_height() with dm vs cm auto-detection
            height = (raw_h * 0.1) if raw_h <= 30.0 else (raw_h / 100.0)
            height = float(np.clip(height, 0.1, 2.5))
        
        # Update dead-reckoning position in Cage Corridor Frame
        now = time.time()
        dt = max(1e-3, min(0.2, now - self.last_time))
        self.last_time = now
        
        self.est_pos[0] += vx_corridor * dt
        self.est_pos[1] += vy_corridor * dt
        
        # DJI Mission Pad Ground-Truth Localization Snap:
        # When a Mission Pad is visible in bottom camera, snap position to ground truth (cancels all drift)
        if self.use_pad:
            try:
                mid = self.tello.get_mission_pad_id()
                if mid > 0:
                    pad_x_m = float(self.tello.get_mission_pad_distance_x()) / 100.0
                    pad_y_m = float(self.tello.get_mission_pad_distance_y()) / 100.0
                    pad_z_m = float(self.tello.get_mission_pad_distance_z()) / 100.0
                    self.last_pad_info = (mid, pad_x_m, pad_y_m, pad_z_m)
                    
                    if self.pad_pos is not None:
                        ref_x, ref_y = self.pad_pos[0], self.pad_pos[1]
                    elif self.pad_at == 'origin':
                        ref_x, ref_y = 0.0, 0.0
                    else: # 'target'
                        ref_x, ref_y = self.target[0], self.target[1]
                        
                    self.est_pos[0] = ref_x + pad_x_m
                    self.est_pos[1] = ref_y + pad_y_m
                else:
                    self.last_pad_info = (-1, 0.0, 0.0, 0.0)
            except Exception:
                pass
        
        # Relative vector to target
        dx = self.target[0] - self.est_pos[0]
        dy = self.target[1] - self.est_pos[1]
        dz = self.target[2] - height
        dist = float(np.sqrt(dx**2 + dy**2 + dz**2))
        
        # Transform delta into Body Frame
        dx_body = float(dx * np.cos(yaw) + dy * np.sin(yaw))
        dy_body = float(-dx * np.sin(yaw) + dy * np.cos(yaw))
        
        # Relative heading wrapped to [-pi, pi]
        heading = float((np.arctan2(dy, dx) - yaw + np.pi) % (2 * np.pi) - np.pi)
        
        state_15 = np.array([
            roll, pitch, yaw,
            vx_body, vy_body, vz,
            height,
            self.est_pos[0], self.est_pos[1],
            dx_body, dy_body, dz, dist, heading, 0.0
        ], dtype=np.float32)
        
        return state_15, dist, heading, height, rel_yaw_deg

    def init_buffers(self):
        """ Pre-fills the 4-frame FIFO buffer with initial stationary readings. """
        depth_map, _, _ = self.depth_worker.get_latest()
        state_15, _, _, _, _ = self.get_kinematics_15d()
        depth_map = depth_map.reshape(48, 64).astype(np.float32)
        state_15 = state_15.reshape(15).astype(np.float32)
        self.depth_buffer.clear()
        self.state_buffer.clear()
        for _ in range(4):
            self.depth_buffer.append(depth_map.copy())
            self.state_buffer.append(state_15.copy())

    def create_hud(self, fpv_frame, depth_vis, dist, height, action, rc_cmds, dist_obs, status, est_pos):
        """ Assembles a high-visibility dual-view telemetry HUD for real-time flight monitoring. """
        h, w = 360, 480
        fpv_res = cv2.resize(fpv_frame, (w, h)) if fpv_frame is not None else np.zeros((h, w, 3), dtype=np.uint8)
        
        if depth_vis is not None:
            depth_res = cv2.resize(depth_vis, (w, h))
        else:
            depth_res = np.zeros((h, w, 3), dtype=np.uint8)
            
        # Draw central targeting crosshair on FPV
        cv2.drawMarker(fpv_res, (w//2, h//2), (0, 255, 0), cv2.MARKER_CROSS, 24, 2)
        
        # Combine side-by-side
        top_view = np.hstack([fpv_res, depth_res])
        
        # Bottom telemetry dashboard
        dash = np.zeros((140, w * 2, 3), dtype=np.uint8)
        color = (0, 255, 0) if status == "NAVIGATING" else ((0, 165, 255) if status == "SLOWING" else (0, 255, 255))
        
        cv2.putText(dash, f"STATUS: {status}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        cv2.putText(dash, f"Target Dist: {dist:.2f}m", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(dash, f"Pos: ({est_pos[0]:+.2f}, {est_pos[1]:+.2f})m", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(dash, f"VPS Raw: ({self.last_vps[0]:+.0f}, {self.last_vps[1]:+.0f}) {self.vps_unit}/s", (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
        
        cv2.putText(dash, f"Alt: {height:.2f}m (tof={int(self.last_raw_tof)}cm)  Obs: {dist_obs:.2f}m", (340, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(dash, f"RC Output     : lr={rc_cmds[0]} fb={rc_cmds[1]} ud={rc_cmds[2]} yaw={rc_cmds[3]}", (340, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
        
        # Mission Pad Status Badge
        if self.use_pad:
            if self.last_pad_info[0] > 0:
                pad_txt = f"PAD #{self.last_pad_info[0]}: LOCK ({self.last_pad_info[1]:+.2f}, {self.last_pad_info[2]:+.2f})m"
                pad_color = (0, 255, 0)
            else:
                pad_txt = "PAD: SEARCHING..."
                pad_color = (0, 255, 255)
        else:
            pad_txt = "PAD: OFF (VPS Only)"
            pad_color = (150, 150, 150)
        cv2.putText(dash, pad_txt, (690, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, pad_color, 2 if self.last_pad_info[0] > 0 else 1)
        
        cv2.putText(dash, "[SPACE/Q] Land / E-STOP", (730, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return np.vstack([top_view, dash])

    def run(self):
        self.connect()
        self.init_buffers()
        
        print("\n" + "=" * 60)
        print("  PRESS ENTER TO TAKEOFF  |  PRESS CTRL+C TO CANCEL")
        print("=" * 60)
        input("Ready? Press [ENTER] to launch...")
        
        # Check pre-launch vitals right before takeoff
        try:
            bat = self.tello.get_battery()
            temp = self.tello.get_highest_temperature()
            print(f"[TELLO] Pre-launch Vitals: Battery: {bat}% | Temp: {temp}°C")
            if bat < 15:
                print(f"[ERROR] Battery is critically low ({bat}%)! Takeoff blocked by DJI safety firmware.")
                return
            if temp >= 85:
                print(f"[ERROR] Drone has overheated ({temp}°C >= 85°C)! Takeoff blocked by DJI safety firmware.")
                print("        Please power off Tello for 1 minute to cool down before flying.")
                return
        except Exception:
            pass

        print("[FLIGHT] Executing autonomous takeoff...")
        try:
            self.tello.takeoff()
        except Exception as e:
            print("\n" + "!" * 70)
            print(f"[ERROR] Takeoff rejected by Tello firmware: {e}")
            try:
                bat = self.tello.get_battery()
                temp = self.tello.get_highest_temperature()
                pitch = self.tello.get_pitch()
                roll = self.tello.get_roll()
                print(f"[DIAGNOSTICS] Battery: {bat}% | Temp: {temp}°C | Pitch: {pitch}° | Roll: {roll}°")
                if temp >= 85:
                    print("-> ROOT CAUSE: OVERHEATING. Tello sat idle too long on the floor without propeller airflow.")
                    print("   FIX: Turn off Tello, wait 60s to cool down, then restart.")
                elif bat < 20:
                    print("-> ROOT CAUSE: LOW BATTERY. Tello firmware blocks takeoff below 15-20%.")
                    print("   FIX: Swap or charge the battery.")
                elif abs(pitch) > 10 or abs(roll) > 10:
                    print("-> ROOT CAUSE: UNEVEN SURFACE. Drone is tilted on the ground.")
                    print("   FIX: Place drone on a flat, level surface.")
                else:
                    print("-> ROOT CAUSE: PREVIOUS FLIGHT LOCKOUT / IMU FAULT.")
                    print("   FIX: Power-cycle the drone (turn off, turn on) to clear the error state.")
            except Exception:
                pass
            print("!" * 70 + "\n")
            raise e
        time.sleep(1.5)
        
        # CALIBRATE TAKEOFF CORRIDOR HEADING:
        # Zero the yaw reference to the exact direction the drone faces in the cage after takeoff
        self.takeoff_yaw_deg = float(self.tello.get_yaw())
        self.est_pos = np.zeros(2, dtype=np.float32)
        print(f"[TELLO] Calibrated cage corridor heading reference: {self.takeoff_yaw_deg:.1f}°")
        
        # Climb/descend to target altitude if needed (using accurate ToF ground clearance)
        init_tof = float(self.tello.get_distance_tof())
        init_h = float(self.tello.get_height())
        if 10.0 <= init_tof <= 300.0:
            cur_h_cm = int(init_tof)
        else:
            cur_h_cm = int(init_h * 10.0 if init_h <= 30.0 else init_h)
            
        target_h_cm = int(self.target[2] * 100.0)
        print(f"[TELLO] Takeoff altitude: {cur_h_cm}cm (tof={init_tof:.0f}cm, h={init_h:.0f}) | Target: {target_h_cm}cm")
        if cur_h_cm < (target_h_cm - 15):
            climb = int(np.clip(target_h_cm - cur_h_cm, 20, 50))
            print(f"[FLIGHT] Adjusting altitude to {target_h_cm}cm (climbing +{climb}cm)...")
            self.tello.move_up(climb)
            time.sleep(0.5)
            
        # Re-fill 4-frame FIFO buffer with clean airborne observations
        self.init_buffers()
        self.last_time = time.time()
        flight_start = time.time()
        step_count = 0
        print("[FLIGHT] Drone airborne. Neural navigation policy ACTIVE!")
        
        try:
            while True:
                # 1. Perception & Kinematics Acquisition
                depth_map, depth_vis, dist_obs = self.depth_worker.get_latest()
                state_15, dist_to_target, heading_err, height, rel_yaw_deg = self.get_kinematics_15d()
                
                # Update 4-frame FIFO stacks
                self.depth_buffer.append(depth_map.reshape(48, 64).astype(np.float32))
                self.state_buffer.append(state_15.reshape(15).astype(np.float32))
                
                # Assemble batched observation dictionary for SB3
                stacked_depth = np.stack(list(self.depth_buffer), axis=0).astype(np.float32) # Strictly (4, 48, 64)
                if stacked_depth.ndim == 4 and stacked_depth.shape[1] == 1:
                    stacked_depth = np.squeeze(stacked_depth, axis=1)
                stacked_state = np.concatenate(list(self.state_buffer), axis=0).astype(np.float32) # Strictly (60,)
                
                obs_dict = {
                    "depth": stacked_depth,
                    "state": stacked_state
                }
                
                # 2. PPO Policy Inference
                action, _ = self.model.predict(obs_dict, deterministic=True)
                lr_act, fb_act, ud_act, yaw_act = action
                
                # 3. Yaw Control: Neural Policy or Autopilot Auto-Yaw
                if self.policy_yaw:
                    # Neural policy directly controls yaw (action[3] in [-1, 1])
                    # In Tello SDK: +yaw is CW (turn right), -yaw is CCW (turn left)
                    yaw_cmd = self.map_actuator(yaw_act, max_cmd=20, deadband=5)
                else:
                    # Autopilot Auto-Yaw Tracking:
                    # Smoothly steers the drone's nose and camera to directly face the target waypoint
                    dx = self.target[0] - self.est_pos[0]
                    dy = self.target[1] - self.est_pos[1]
                    if dist_to_target > 0.35:
                        target_yaw_deg = -float(np.degrees(np.arctan2(dy, dx)))  # In Tello, CW (Right) is positive
                        yaw_err_deg = (target_yaw_deg - rel_yaw_deg + 180.0) % 360.0 - 180.0
                        yaw_cmd = int(np.clip(yaw_err_deg * 0.6, -15, 15))  # Smooth, gentle turning (max +-15)
                    else:
                        yaw_cmd = 0
                
                # 4. Cruising Altitude Hold + Vertical Dodge
                z_err = self.target[2] - height
                vz_cmd = np.clip(1.5 * z_err + ud_act * 0.4, -0.6, 0.6)
                ud_cmd = int(np.clip(vz_cmd * 50.0, -35, 35))
                
                status = "NAVIGATING"
                
                # 5. Physical RC Command Mapping (Matched Actuator Dynamics)
                # In simulation: vy_body = -lr * 1.25 (lateral velocity has 1.25x gain)
                max_lat_cmd = min(30, int(self.max_speed_cm * 1.25))
                fb_cmd = self.map_actuator(fb_act, max_cmd=self.max_speed_cm, deadband=6)
                lr_cmd = self.map_actuator(lr_act, max_cmd=max_lat_cmd, deadband=6)
                
                # Send to flight controller
                self.tello.send_rc_control(lr_cmd, fb_cmd, ud_cmd, yaw_cmd)
                rc_cmds = [lr_cmd, fb_cmd, ud_cmd, yaw_cmd]
                self.last_rc_cmds = rc_cmds
                
                # Periodic terminal telemetry
                step_count += 1
                if step_count % 15 == 0:
                    pad_str = f"PAD:#{self.last_pad_info[0]}({self.last_pad_info[1]:+.2f}m)" if (self.use_pad and self.last_pad_info[0] > 0) else ("PAD:SEARCH" if self.use_pad else "PAD:OFF")
                    print(f"[NAV] Pos: ({self.est_pos[0]:+.2f}, {self.est_pos[1]:+.2f})m | Alt: {height:.2f}m (tof={self.last_raw_tof:.0f}cm) | {pad_str} | VPS: (vx={state_15[3]:+.2f}, vy={state_15[4]:+.2f})m/s [raw={int(self.last_vps[0])},{int(self.last_vps[1])}] | Dist: {dist_to_target:.2f}m | RC: (lr={lr_cmd}, fb={fb_cmd}, ud={ud_cmd}, yaw={yaw_cmd})")
                
                # 7. Render Telemetry HUD
                raw_frame = self.tello.get_frame_read().frame
                hud = self.create_hud(raw_frame, depth_vis, dist_to_target, height, action, rc_cmds, dist_obs, status, self.est_pos)
                cv2.imshow("Project ICARUS - Live Drone Autonomous Telemetry", hud)
                
                # Listen for keyboard interrupts
                key = cv2.waitKey(20) & 0xFF
                if key == 32 or key == ord('q'): # Spacebar or 'q' = Emergency E-Stop
                    print("\n[SAFETY] EMERGENCY STOP ACTIVATED BY USER!")
                    break
                    
                # 8. Mission Completion
                if dist_to_target < 0.20: # 20cm arrival bullseye (aligned with training arrival threshold)
                    print(f"\n[MISSION COMPLETE] Target Reached! Final distance: {dist_to_target:.2f}m")
                    break
                    
                if abs(state_15[0]) > np.radians(35) or abs(state_15[1]) > np.radians(35):
                    print("\n[SAFETY] Excessive tilt angle detected (>35 deg)! Aborting.")
                    break
                    
        except KeyboardInterrupt:
            print("\n[FLIGHT] Keyboard Interrupt detected!")
        except Exception as e:
            print(f"\n[ERROR] Runtime exception: {e}")
        finally:
            print("[FLIGHT] Safely terminating flight...")
            try:
                self.tello.send_rc_control(0, 0, 0, 0)
                time.sleep(0.3)
                self.tello.land()
            except Exception as e:
                print(f"[WARN] Standard land ACK timeout: {e}. Sending direct land command...")
                try:
                    self.tello.send_command_without_return("land")
                except Exception:
                    pass
            if self.depth_worker is not None:
                self.depth_worker.stop()
            if self.use_pad:
                try:
                    self.tello.disable_mission_pads()
                except Exception:
                    pass
            try:
                self.tello.streamoff()
                self.tello.end()
            except Exception:
                pass
            cv2.destroyAllWindows()
            print("[TELLO] Landed and disconnected cleanly.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deploy Project ICARUS on Physical DJI Tello Drone')
    parser.add_argument('--model', type=str, default='results/navigate/stage_3_icarus/best_model.zip', help='Path to best_model.zip')
    parser.add_argument('--target', type=float, nargs=3, default=[1.0, 0.0, 0.8], help='Target X Y Z relative to takeoff point (meters)')
    parser.add_argument('--speed', type=int, default=30, help='Max cruising speed in cm/s (default: 30 cm/s for 2x2x2m cage)')
    parser.add_argument('--vps_unit', type=str, choices=['dm', 'cm'], default='dm', help='VPS velocity unit from SDK telemetry (default: dm for RoboMaster TT SDK 3.0, cm for standard Tello)')
    parser.add_argument('--vps_x_sign', type=int, choices=[1, -1], default=-1, help='VPS forward speed vgx sign multiplier (default: -1 for RoboMaster TT where forward movement reports negative vgx)')
    parser.add_argument('--vps_y_sign', type=int, choices=[1, -1], default=1, help='VPS lateral speed vgy sign multiplier (default: 1 because raw_vgy < 0 when moving right)')
    parser.add_argument('--policy_yaw', action='store_true', help='Use neural policy directly for yaw control instead of autopilot auto-yaw')
    parser.add_argument('--use_pad', action='store_true', help='Enable DJI Mission Pad detection and localization fusion')
    parser.add_argument('--pad_at', type=str, choices=['target', 'origin'], default='target', help='Physical location of the mission pad on cage floor (default: target)')
    parser.add_argument('--pad_pos', type=float, nargs=2, default=None, help='Explicit (X, Y) world coordinates of the mission pad in meters (overrides --pad_at)')
    
    args = parser.parse_args()
    agent = RealTelloIcarusAgent(
        model_path=args.model,
        target_pos=args.target,
        max_speed_cm=args.speed,
        vps_unit=args.vps_unit,
        vps_x_sign=args.vps_x_sign,
        vps_y_sign=args.vps_y_sign,
        policy_yaw=args.policy_yaw,
        use_pad=args.use_pad,
        pad_at=args.pad_at,
        pad_pos=args.pad_pos
    )
    agent.run()
