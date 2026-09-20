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
                 act: ActionType=ActionType.VEL
                 ):
                 
        self.TARGET_POS = np.array([2.0, 2.0, 1.0])
        self.ARRIVAL_THRESHOLD = 0.15 # 15cm
        self.EPISODE_LEN_SEC = 30
        
        # Will be updated during resets
        self.prev_distance = 0.0
        self.est_pos = np.zeros(2)
        
        # Overriding IMG_RES for depth camera to match plan (W=64, H=48)
        self.IMG_RES = np.array([64, 48])
        
        # Action delay buffer for domain randomization
        self.action_buffer = deque(maxlen=4)
        for _ in range(4):
            self.action_buffer.append(np.zeros(4))
            
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
        depth_linear = far * near / (far - (far - near) * dep)
        
        # Normalize depth to 0-1 range, clipping at 5 meters
        MAX_DEPTH = 5.0
        dep_normalized = np.clip(depth_linear, 0, MAX_DEPTH) / MAX_DEPTH
        dep_tensor = np.expand_dims(dep_normalized, axis=0).astype(np.float32) # (1, 48, 64)
        
        # State vector
        state = self._getDroneStateVector(0)
        
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
        
        return {"depth": dep_tensor, "state": obs_state}
        
    def _preprocessAction(self, action):
        """ Maps [-1, 1] 4D action to motor RPMs via PID tracking. """
        # Handle dict/vecenv formatting if needed
        # Since it's a single agent, it might come as shape (4,) or (1, 4)
        target = action.flatten()
        
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
        
        target_vel = self.SPEED_LIMIT * np.array([vx_world, vy_world, vz_world])
        
        # Yaw rate: Tello positive is CW (negative Z rotation in right-hand rule)
        # We'll map [-1, 1] to a max rotation rate (e.g. 1.0 rad/s)
        # So target yaw = current yaw + (-yaw_rate * MAX_RATE * dt)
        target_yaw = yaw - yaw_rate * 1.0 * self.CTRL_TIMESTEP
        
        rpm, _, _ = self.ctrl[0].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                cur_pos=state[0:3],
                                                cur_quat=state[3:7],
                                                cur_vel=state[10:13],
                                                cur_ang_vel=state[13:16],
                                                target_pos=state[0:3], # don't care about position tracking
                                                target_rpy=np.array([0, 0, target_yaw]),
                                                target_vel=target_vel
                                                )
        return np.array([rpm])
        
    def _addObstacles(self):
        """ Overrides BaseRLAviary to add randomized obstacles. """
        self.obstacle_ids = []
        n_obstacles = np.random.randint(6, 10)
        for _ in range(n_obstacles):
            # Random position within bounds, avoiding start (0,0) and target
            while True:
                pos = [np.random.uniform(-3, 3), np.random.uniform(-3, 3), 0]
                dist_start = np.linalg.norm(np.array(pos) - np.array([0,0,0]))
                dist_target = np.linalg.norm(np.array(pos) - self.TARGET_POS)
                if dist_start > 0.8 and dist_target > 0.8:
                    break
                    
            if np.random.random() > 0.5:
                # Cylinder
                shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.15, height=2.0, physicsClientId=self.CLIENT)
                visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.15, length=2.0, rgbaColor=[0.8, 0.2, 0.1, 1], physicsClientId=self.CLIENT)
                pos[2] = 1.0 # center of 2.0 height
            else:
                # Box
                shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 1.0], physicsClientId=self.CLIENT)
                visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 1.0], rgbaColor=[0.9, 0.4, 0.1, 1], physicsClientId=self.CLIENT)
                pos[2] = 1.0 # center of 2.0 height
                
            obs_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape,
                                       baseVisualShapeIndex=visual, basePosition=pos,
                                       physicsClientId=self.CLIENT)
            self.obstacle_ids.append(obs_id)
            
        # Draw a clean 3D green circle (wireframe sphere) using debug lines.
        # This perfectly outlines the 0.15m arrival threshold and is completely invisible to the depth camera!
        radius = 0.15
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
        
        # REMOVED: reward = -distance * 0.1 (This caused the drone to suicide to avoid accumulating distance penalties)
        reward = progress * 50.0
        reward += max(0, 1.0 - distance) * 0.1
        
        # Time penalty to create urgency (stop it from hovering forever)
        reward -= 0.05
        
        # We removed the dense tilt penalty because drones MUST pitch to move forward!
        # Penalizing pitch just teaches it to stay perfectly flat and hover in place.
        
        if self._check_collision():
            reward -= 100.0
            
        if abs(pos[0]) > 4.0 or abs(pos[1]) > 4.0 or pos[2] > 3.0:
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
        if abs(state[0]) > 4.0 or abs(state[1]) > 4.0 or state[2] > 3.0:
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
        self.TARGET_POS = np.array([
            np.random.uniform(1.0, 3.0) * np.random.choice([-1, 1]),
            np.random.uniform(1.0, 3.0) * np.random.choice([-1, 1]),
            np.random.uniform(0.5, 1.5)
        ])
        
        # Reset estimates
        self.est_pos = np.zeros(2)
        
        obs, info = super().reset(seed=seed, options=options)
        
        # Initialize prev distance
        state = self._getDroneStateVector(0)
        self.prev_distance = np.linalg.norm(self.TARGET_POS - state[0:3])
        
        # Fill action delay buffer
        delay_steps = np.random.randint(1, 4)
        self.action_buffer = deque(maxlen=delay_steps)
        for _ in range(delay_steps):
            self.action_buffer.append(np.zeros(4))
            
        return obs, info
