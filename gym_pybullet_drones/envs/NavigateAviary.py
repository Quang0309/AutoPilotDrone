import numpy as np
import pybullet as p
from gymnasium import spaces
from collections import deque
import random
import os

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class NavigateAviary(BaseRLAviary):
    """
    Tello-compatible navigation environment with obstacles and vision.
    Action space: [lr, fb, ud, yaw] in [-1, 1], matches Tello's send_rc_control.
    Observation space: dict with 'depth' (1, 48, 64) and 'state' (15,).
    """
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 24,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.ALL,
                 act: ActionType=ActionType.VEL,
                 difficulty: str='hard'
                 ):
                 
        self.DIFFICULTY = difficulty  # 'easy', 'medium', or 'hard'
        self.TARGET_POS = np.array([2.0, 2.0, 1.0])
        
        if difficulty == 'easy':
            self.ARRIVAL_THRESHOLD = 0.3
            self.EPISODE_LEN_SEC = 15
            self.BOUND_LIMIT_XY = 4.0
        elif difficulty == 'medium':
            self.ARRIVAL_THRESHOLD = 0.3   # Same as easy — keep success achievable
            self.EPISODE_LEN_SEC = 20
            self.BOUND_LIMIT_XY = 4.0
        else:  # hard
            self.ARRIVAL_THRESHOLD = 0.25
            self.EPISODE_LEN_SEC = 30       # 30s allows comfortable long-distance flight (up to 4.8m)
            self.BOUND_LIMIT_XY = 6.0       # Expands flight arena from 8x8m to 12x12m
        
        # Will be updated during resets
        self.prev_distance = 0.0
        self.prev_dz = 0.0
        self.est_pos = np.zeros(2)
        
        # Overriding IMG_RES for depth camera to match plan (W=64, H=48)
        self.IMG_RES = np.array([64, 48])
        
        # Action delay buffer for domain randomization
        self.action_buffer = deque(maxlen=4)
        for _ in range(4):
            self.action_buffer.append(np.zeros(4))
            
        if initial_xyzs is None:
            initial_xyzs = np.array([[0.0, 0.0, 1.0]])
            
        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )
        
        # Use a slightly smaller max speed for Tello-like behavior
        self.SPEED_LIMIT = 1.0 # m/s
        
    def _actionSpace(self):
        """ Returns the action space. Matches Tello [-1, 1] 4D commands. """
        return spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        
    def _observationSpace(self):
        """ Custom dict observation space. """
        if self.OBS_TYPE == ObservationType.KIN:
            return spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        else:
            return spaces.Dict({
                "depth": spaces.Box(low=0.0, high=1.0, shape=(1, 48, 64), dtype=np.float32),
                "state": spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
            })

    def _computeObs(self):
        """ Computes current observation (depth + state). """
        # Render depth
        rgb, dep, seg = self._getDroneImages(0, segmentation=False)
        
        # Convert PyBullet's non-linear OpenGL depth buffer to linear distance in meters
        near = self.L
        far = 1000.0
        dep_clamped = np.clip(np.nan_to_num(dep, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        denom = np.maximum(far - (far - near) * dep_clamped, 1e-4)
        depth_linear = far * near / denom
        
        # Normalize depth to 0-1 range, clipping at 5 meters
        MAX_DEPTH = 5.0
        dep_normalized = np.clip(depth_linear, 0.0, MAX_DEPTH) / MAX_DEPTH
        dep_tensor = np.expand_dims(dep_normalized, axis=0).astype(np.float32) # (1, 48, 64)
        dep_tensor = np.nan_to_num(dep_tensor, nan=1.0, posinf=1.0, neginf=0.0)
        
        # [NEW] Blindfold the CNN in Easy Mode to guarantee it doesn't inject random noise
        # Using 0.01 instead of exactly 0.0 to bypass a known PyTorch MPS bug where exact zero tensors cause NaNs in Conv2d
        if self.DIFFICULTY == 'easy':
            dep_tensor = np.full_like(dep_tensor, 0.01)
        
        # State vector (sanitized against physics engine numerical spikes)
        state = self._getDroneStateVector(0)
        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Extract ground truth state
        pos = state[0:3]
        rpy = state[7:10]
        vel = state[10:13]
        
        # Apply Observation Noise
        attitude_noise = np.random.normal(0, np.radians(1.0), 3) # 1 degree noise
        velocity_noise = np.random.normal(0, 0.02, 3) # 2cm/s noise
        height_noise = np.random.normal(0, 0.01) # 1cm noise
        
        noisy_rpy = rpy + attitude_noise
        noisy_vel = vel + velocity_noise
        noisy_height = pos[2] + height_noise
        
        # Use real position + small noise (simulates Visual-Inertial Odometry)
        # This is noisy but does NOT accumulate drift over time like dead reckoning
        pos_noise_xy = np.random.normal(0, 0.05, 2)  # 5cm noise on XY
        noisy_pos_xy = pos[0:2] + pos_noise_xy
        
        # Target relative info based on noisy real position
        dx = self.TARGET_POS[0] - noisy_pos_xy[0]
        dy = self.TARGET_POS[1] - noisy_pos_xy[1]
        dz = self.TARGET_POS[2] - noisy_height
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        
        yaw = noisy_rpy[2]
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        
        # Convert target delta to Body Frame
        dx_body = dx * cos_y + dy * sin_y
        dy_body = -dx * sin_y + dy * cos_y
        
        # Convert velocity to Body Frame
        vx_body = noisy_vel[0] * cos_y + noisy_vel[1] * sin_y
        vy_body = -noisy_vel[0] * sin_y + noisy_vel[1] * cos_y
        
        # Heading wrapped to [-pi, pi]
        heading = np.arctan2(dy, dx) - yaw
        heading = (heading + np.pi) % (2 * np.pi) - np.pi
        
        obs_state = np.array([
            noisy_rpy[0], noisy_rpy[1], yaw,
            vx_body, vy_body, noisy_vel[2],
            noisy_height,
            noisy_pos_xy[0], noisy_pos_xy[1],
            dx_body, dy_body, dz, dist, heading, 0.0
        ], dtype=np.float32)
        obs_state = np.nan_to_num(obs_state, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if self.OBS_TYPE == ObservationType.KIN:
            return obs_state
        return {"depth": dep_tensor, "state": obs_state}
        
    def _preprocessAction(self, action):
        """ Maps [-1, 1] 4D action to motor RPMs via PID tracking. """
        # Handle dict/vecenv formatting if needed
        # Since it's a single agent, it might come as shape (4,) or (1, 4)
        target = np.clip(np.nan_to_num(action.flatten(), nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
        
        # Action Delay
        self.action_buffer.append(target)
        delayed_action = self.action_buffer[0] # Use oldest action in buffer
        
        lr, fb, ud, yaw_rate = delayed_action
        
        state = self._getDroneStateVector(0)
        yaw = state[9]
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        
        # Body frame to world frame (PyBullet: X is forward, Y is left)
        # fb -> X body, -lr -> Y body
        vx_body = fb
        vy_body = -lr
        
        vx_world = vx_body * cos_yaw - vy_body * sin_yaw
        vy_world = vx_body * sin_yaw + vy_body * cos_yaw
        vz_world = ud
        
        # Terminal approach velocity & yaw governor:
        # Smoothly scales speed and yaw responsiveness down when close to the target,
        # preventing violent high-speed overshoots, gyro swing, and tilt oscillations.
        dist_to_target = np.linalg.norm(self.TARGET_POS - state[0:3])
        speed_scale = float(np.clip(dist_to_target / 0.8, 0.25, 1.0))
        yaw_scale = float(np.clip(dist_to_target / 0.5, 0.35, 1.0))
        
        target_vel = (self.SPEED_LIMIT * speed_scale) * np.array([vx_world, vy_world, vz_world])
        
        # Yaw rate: agile at long range (3.0 rad/s), gentle near target (scaled down to ~1.0 rad/s)
        target_yaw = yaw - yaw_rate * (3.0 * yaw_scale) * self.CTRL_TIMESTEP
        
        rpm, _, _ = self.ctrl[0].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                cur_pos=state[0:3],
                                                cur_quat=state[3:7],
                                                cur_vel=state[10:13],
                                                cur_ang_vel=state[13:16],
                                                target_pos=state[0:3] + target_vel * 0.5, # Carrot on a stick for proper PID P-term tracking
                                                target_rpy=np.array([0, 0, target_yaw]),
                                                target_vel=target_vel
                                                )
        return np.array([rpm])
        
    def _addObstacles(self):
        """ Overrides BaseRLAviary to add randomized obstacles. """
        self.obstacle_ids = []
        if self.DIFFICULTY == 'easy':
            obstacle_positions = []
        elif self.DIFFICULTY == 'medium':
            n_obstacles = np.random.randint(2, 5)   # 2-4 obstacles: gentle CNN training
            obstacle_positions = []
            for _ in range(n_obstacles):
                while True:
                    pos = [np.random.uniform(-3, 3), np.random.uniform(-3, 3), 1.0]
                    dist_start = np.linalg.norm(np.array(pos[:2]) - np.array([0, 0]))
                    dist_target = np.linalg.norm(np.array(pos[:2]) - self.TARGET_POS[:2])
                    if dist_start > 0.8 and dist_target > 0.8:
                        obstacle_positions.append(pos)
                        break
        else:  # hard: well-spaced, meaningful slalom obstacles along the flight corridor
            start_xy = np.array([0.0, 0.0])
            target_xy = self.TARGET_POS[0:2]
            D = np.linalg.norm(target_xy - start_xy)
            u = (target_xy - start_xy) / max(D, 1e-4)   # Unit vector along line of sight
            u_perp = np.array([-u[1], u[0]])             # Perpendicular normal vector
            
            obstacle_positions = []
            
            # Staggered slalom layout (3 obstacles, spaced ~1.3m to 1.5m apart)
            # Each obstacle directly interrupts straight flight, but leaves at least 1.5m - 2.0m of wide, open navigable airspace on the sides!
            side1 = np.random.choice([-1, 1])
            
            # 1. Obstacle 1: at ~35% distance (~1.5m from start), offset slightly to one side
            pos1 = start_xy + 0.35 * D * u + (side1 * 0.25) * u_perp
            obstacle_positions.append([pos1[0], pos1[1], 1.0])
            
            # 2. Obstacle 2: at ~65% distance (~2.9m from start), offset to the opposite side
            pos2 = start_xy + 0.65 * D * u - (side1 * 0.30) * u_perp
            obstacle_positions.append([pos2[0], pos2[1], 1.0])
            
            # 3. Obstacle 3: at ~82% distance (~3.7m from start), offset near target zone
            pos3 = start_xy + 0.82 * D * u + np.random.uniform(-0.15, 0.15) * u_perp
            obstacle_positions.append([pos3[0], pos3[1], 1.0])

        for pos in obstacle_positions:
            if np.random.random() > 0.5:
                # Cylinder
                shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.15, height=2.0, physicsClientId=self.CLIENT)
                visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.15, length=2.0, rgbaColor=[0.8, 0.2, 0.1, 1], physicsClientId=self.CLIENT)
            else:
                # Box
                shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 1.0], physicsClientId=self.CLIENT)
                visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 1.0], rgbaColor=[0.9, 0.4, 0.1, 1], physicsClientId=self.CLIENT)
                
            obs_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape,
                                       baseVisualShapeIndex=visual, basePosition=pos,
                                       physicsClientId=self.CLIENT)
            self.obstacle_ids.append(obs_id)
            
        # Draw a clean 3D green circle (wireframe sphere) using debug lines.
        # Outlines the exact arrival threshold and is completely invisible to the depth camera!
        radius = self.ARRIVAL_THRESHOLD
        segments = 16
        for i in range(segments):
            angle1 = 2 * np.pi * i / segments
            angle2 = 2 * np.pi * (i + 1) / segments
            
            # XY Plane Circle
            p1_xy = self.TARGET_POS + np.array([np.cos(angle1)*radius, np.sin(angle1)*radius, 0])
            p2_xy = self.TARGET_POS + np.array([np.cos(angle2)*radius, np.sin(angle2)*radius, 0])
            p.addUserDebugLine(p1_xy, p2_xy, lineColorRGB=[0.1, 1.0, 0.1], lineWidth=3, physicsClientId=self.CLIENT)
            
            # XZ Plane Circle
            p1_xz = self.TARGET_POS + np.array([np.cos(angle1)*radius, 0, np.sin(angle1)*radius])
            p2_xz = self.TARGET_POS + np.array([np.cos(angle2)*radius, 0, np.sin(angle2)*radius])
            p.addUserDebugLine(p1_xz, p2_xz, lineColorRGB=[0.1, 1.0, 0.1], lineWidth=3, physicsClientId=self.CLIENT)
            
            # YZ Plane Circle
            p1_yz = self.TARGET_POS + np.array([0, np.cos(angle1)*radius, np.sin(angle1)*radius])
            p2_yz = self.TARGET_POS + np.array([0, np.cos(angle2)*radius, np.sin(angle2)*radius])
            p.addUserDebugLine(p1_yz, p2_yz, lineColorRGB=[0.1, 1.0, 0.1], lineWidth=3, physicsClientId=self.CLIENT)
        
    def _check_collision(self):
        for obs_id in getattr(self, 'obstacle_ids', []):
            contacts = p.getContactPoints(bodyA=self.DRONE_IDS[0], bodyB=obs_id, physicsClientId=self.CLIENT)
            if contacts is not None and len(contacts) > 0:
                return True
        # Check ground
        contacts = p.getContactPoints(bodyA=self.DRONE_IDS[0], bodyB=self.PLANE_ID, physicsClientId=self.CLIENT)
        if contacts is not None and len(contacts) > 0:
            return True
        return False
        
    def _computeReward(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        distance = np.linalg.norm(self.TARGET_POS - pos)
        progress = self.prev_distance - distance
        
        # 1. 3D distance progress (meters closed in 3D toward target)
        reward = progress * 50.0
        
        # 2. Dedicated Z-axis progress: strongly reward closing the vertical altitude gap!
        dz = abs(self.TARGET_POS[2] - pos[2])
        z_progress = self.prev_dz - dz
        reward += z_progress * 30.0
        self.prev_dz = dz
        
        # 3. Heading alignment: reward pointing camera at the target
        yaw = state[9]
        dx = self.TARGET_POS[0] - pos[0]
        dy = self.TARGET_POS[1] - pos[1]
        heading = np.arctan2(dy, dx) - yaw
        heading = (heading + np.pi) % (2 * np.pi) - np.pi
        reward += np.cos(heading) * 0.20
        
        # 4. Penalize excessive spinning so camera remains stable
        yaw_rate = state[15]
        reward -= abs(yaw_rate) * 0.02
        
        # 5. Time penalty: creates urgency to descend and hit the bullseye rather than hovering
        reward -= 0.06
        
        # 6. Terminal deceleration incentive: discourage rushing at >0.4 m/s when close to target (<0.6m)
        speed = np.linalg.norm(state[10:13])
        if distance < 0.6 and speed > 0.4:
            reward -= (speed - 0.4) * 0.5
        
        # We removed the dense tilt penalty because drones MUST pitch to move forward!
        # Penalizing pitch just teaches it to stay perfectly flat and hover in place.
        
        if self._check_collision():
            reward -= 100.0
            
        if abs(pos[0]) > self.BOUND_LIMIT_XY or abs(pos[1]) > self.BOUND_LIMIT_XY or pos[2] > 3.0:
            reward -= 100.0
            
        if abs(state[7]) > 1.5 or abs(state[8]) > 1.5:
            reward -= 100.0
            
        if distance < self.ARRIVAL_THRESHOLD:
            reward += 100.0
            
        self.prev_distance = distance
        return float(reward)
        
    def _computeTerminated(self):
        state = self._getDroneStateVector(0)
        distance = np.linalg.norm(self.TARGET_POS - state[0:3])
        return bool(distance < self.ARRIVAL_THRESHOLD)
        
    def _computeTruncated(self):
        state = self._getDroneStateVector(0)
        if self._check_collision():
            return True
        if abs(state[0]) > self.BOUND_LIMIT_XY or abs(state[1]) > self.BOUND_LIMIT_XY or state[2] > 3.0:
            return True
        if abs(state[7]) > 1.5 or abs(state[8]) > 1.5:
            return True
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        return False
        
    def _computeInfo(self):
        state = self._getDroneStateVector(0)
        distance = np.linalg.norm(self.TARGET_POS - state[0:3])
        success = bool(distance < self.ARRIVAL_THRESHOLD)
        return {"distance": float(distance), "collision": self._check_collision(), "success": success}

    def reset(self, seed=None, options=None):
        # Randomize Target Position
        if self.DIFFICULTY == 'easy':
            # Target is closer and simpler
            self.TARGET_POS = np.array([
                np.random.uniform(0.5, 1.5) * np.random.choice([-1, 1]),
                np.random.uniform(0.5, 1.5) * np.random.choice([-1, 1]),
                np.random.uniform(0.8, 1.2)
            ])
        elif self.DIFFICULTY == 'medium':
            # Moderate distances, same as easy range but CNN sees obstacles
            self.TARGET_POS = np.array([
                np.random.uniform(0.5, 2.0) * np.random.choice([-1, 1]),
                np.random.uniform(0.5, 2.0) * np.random.choice([-1, 1]),
                np.random.uniform(0.7, 1.3)
            ])
        else:
            # Hard mode: long-range navigation (guaranteed 3.8m to 4.8m distance across 12x12m space)
            theta = np.random.uniform(0, 2 * np.pi)
            R = np.random.uniform(3.8, 4.8)
            self.TARGET_POS = np.array([
                R * np.cos(theta),
                R * np.sin(theta),
                np.random.uniform(0.8, 1.4)
            ])
        
        # Reset estimates
        self.est_pos = np.zeros(2)
        
        obs, info = super().reset(seed=seed, options=options)
        
        # Initialize prev distance and prev dz
        state = self._getDroneStateVector(0)
        self.prev_distance = np.linalg.norm(self.TARGET_POS - state[0:3])
        self.prev_dz = abs(self.TARGET_POS[2] - state[2])
        
        # Fill action delay buffer
        delay_steps = np.random.randint(1, 4)
        self.action_buffer = deque(maxlen=delay_steps)
        for _ in range(delay_steps):
            self.action_buffer.append(np.zeros(4))
            
        return obs, info
