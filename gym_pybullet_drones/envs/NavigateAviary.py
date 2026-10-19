import numpy as np
import pybullet as p
from gymnasium import spaces
from collections import deque
import random
import os
import cv2
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
        
        # Physical Crazyflie dimensions:
        # Arm length L = 0.0397m + propeller radius = 0.0231m + margin = ~0.08m
        self.DRONE_RADIUS = 0.08
        self.TARGET_RADIUS = 0.25  # Physical 25cm radius target zone
        
        if difficulty == 'easy':
            self.ARRIVAL_THRESHOLD = 0.35
            self.EPISODE_LEN_SEC = 15
            self.BOUND_LIMIT_XY = 1.0
            self.COLLISION_PENALTY = 30.0  # Updated per user request
            self.ARRIVAL_BONUS = 40.0      # Overwhelming incentive to finish the episode
            self.BOUND_PENALTY = 5.0
        elif difficulty == 'medium':
            self.ARRIVAL_THRESHOLD = 0.35
            self.EPISODE_LEN_SEC = 20
            self.BOUND_LIMIT_XY = 1.0
            self.COLLISION_PENALTY = 30.0  # Updated per user request
            self.ARRIVAL_BONUS = 40.0
            self.BOUND_PENALTY = 5.0
        else:  # hard
            # Arrival threshold accounts for drone body radius touching the 0.25m target zone
            self.ARRIVAL_THRESHOLD = self.TARGET_RADIUS + self.DRONE_RADIUS  # 0.33m
            self.EPISODE_LEN_SEC = 30       # 30s allows comfortable flight
            self.BOUND_LIMIT_XY = 1.0       # 2m x 2m cage
            self.COLLISION_PENALTY = 30.0    # Already set for hard mode
            self.ARRIVAL_BONUS = 50.0       # Finishing is overwhelmingly rewarding
            self.BOUND_PENALTY = 10.0
        
        # Will be updated during resets
        self.prev_pos = np.zeros(3)
        self.prev_distance = 0.0
        self.prev_dz = 0.0
        self.est_pos = np.zeros(2)
        self.smoothed_action = np.zeros(4)
        self.net_ids = []
        
        # Overriding IMG_RES for depth camera to match plan (W=64, H=48)
        self.IMG_RES = np.array([64, 48])
        
        # Action delay buffer for domain randomization
        self.action_buffer = deque(maxlen=4)
        for _ in range(4):
            self.action_buffer.append(np.zeros(4))
        self.prev_action = np.zeros(4)
            
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
        
        # MiDaS inverse disparity processing
        disp = 1.0 / (depth_linear + 0.1)
        disp_min, disp_max = np.percentile(disp, 2), np.percentile(disp, 98)
        depth_norm = np.clip((disp_max - disp) / (disp_max - disp_min + 1e-6), 0.0, 1.0)
        # Apply Gaussian blur and small noise
        depth_blur = cv2.GaussianBlur(depth_norm, (3, 3), sigmaX=1.0)
        depth_noisy = depth_blur + np.random.normal(0, 0.03, depth_blur.shape)
        dep_tensor = np.expand_dims(np.clip(depth_noisy, 0.0, 1.0).astype(np.float32), axis=0)  # (1, 48, 64)
        
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
        """Maps [-1, 1] 4D RC action to motor RPMs via DSLPIDControl.

        Action layout matches djitellopy send_rc_control exactly:
            action[0]  lr        left_right_velocity     (-1 = left,    +1 = right)
            action[1]  fb        forward_backward_velocity (-1 = back,  +1 = forward)
            action[2]  ud        up_down_velocity         (-1 = down,   +1 = up)
            action[3]  yaw_rate  yaw_velocity             (-1 = CCW,    +1 = CW)

        Deployment adapter:
            tello.send_rc_control(int(lr*100), int(fb*100), int(ud*100), int(yaw_rate*100))
        """
        target = np.clip(np.nan_to_num(action.flatten(), nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)

        # EMA smoothing to reduce control jerk
        if not hasattr(self, 'smoothed_action') or self.smoothed_action is None:
            self.smoothed_action = target.copy()
        else:
            self.smoothed_action = 0.5 * target + 0.5 * self.smoothed_action

        # Action delay buffer (domain randomisation)
        self.action_buffer.append(self.smoothed_action.copy())
        delayed_action = self.action_buffer[0]

        lr, fb, ud, yaw_rate = delayed_action

        state = self._getDroneStateVector(0)
        yaw = state[9]

        # --- Yaw: fully policy-controlled (matches real Tello yaw_velocity) ---
        # yaw_rate in [-1, 1] → up to ±2 rad/s rotation
        MAX_YAW_RATE = 2.0  # rad/s  (Tello max is ~100°/s ≈ 1.75 rad/s)
        target_yaw = yaw + yaw_rate * MAX_YAW_RATE * self.CTRL_TIMESTEP

        # --- Altitude hold + policy up/down override ---
        z_err = self.TARGET_POS[2] - state[2]
        vz_autopilot = np.clip(1.5 * z_err, -0.6, 0.6)
        vz_world = np.clip(vz_autopilot + ud * 0.4, -0.8, 0.8)

        # --- Horizontal velocity in world frame ---
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        vx_body =  fb          # forward along nose
        vy_body = -lr * 1.25   # lateral (left/right)
        vx_world = vx_body * cos_yaw - vy_body * sin_yaw
        vy_world = vx_body * sin_yaw + vy_body * cos_yaw

        # Terminal approach speed governor
        dist_to_target = np.linalg.norm(self.TARGET_POS - state[0:3])
        speed_scale = float(np.clip(dist_to_target / 0.8, 0.35, 1.0))

        vel_h = np.array([vx_world, vy_world])
        norm_h = np.linalg.norm(vel_h)
        max_h_speed = 0.75 * speed_scale
        if norm_h > max_h_speed and norm_h > 1e-6:
            vel_h = (vel_h / norm_h) * max_h_speed

        target_vel = np.array([vel_h[0], vel_h[1], vz_world * speed_scale])
        clipped_vel = np.clip(target_vel, -self.SPEED_LIMIT, self.SPEED_LIMIT)

        # DSLPIDControl: hover at current pos + apply policy velocity + policy yaw
        rpm, _, _ = self.ctrl[0].computeControl(
            control_timestep=self.CTRL_TIMESTEP,
            cur_pos=state[0:3],
            cur_quat=state[3:7],
            cur_vel=state[10:13],
            cur_ang_vel=state[13:16],
            target_pos=state[0:3],                       # hold position (hover)
            target_rpy=np.array([0.0, 0.0, target_yaw]), # policy-controlled yaw
            target_vel=clipped_vel,                       # policy velocity
        )
        return rpm


        
    @staticmethod
    def _getQuatFromZToVec(v):
        """ Computes quaternion [x, y, z, w] to rotate default Z-axis [0, 0, 1] to target vector v. """
        v = np.array(v, dtype=np.float64)
        norm = np.linalg.norm(v)
        if norm < 1e-6:
            return [0.0, 0.0, 0.0, 1.0]
        v = v / norm
        z = np.array([0.0, 0.0, 1.0])
        dot = np.dot(z, v)
        if dot > 0.999999:
            return [0.0, 0.0, 0.0, 1.0]
        elif dot < -0.999999:
            return [1.0, 0.0, 0.0, 0.0]
        axis = np.cross(z, v)
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.clip(dot, -1.0, 1.0))
        s = np.sin(angle / 2.0)
        c = np.cos(angle / 2.0)
        return [float(axis[0]*s), float(axis[1]*s), float(axis[2]*s), float(c)]

    def _addObstacles(self):
        """ Overrides BaseRLAviary to add randomized 3D obstacles. """
        self.obstacle_ids = []
        self.net_ids = []
        if self.DIFFICULTY == 'easy':
            pass
        elif self.DIFFICULTY == 'medium':
            # Exactly 1 obstacle sitting directly between the drone spawn and the target
            start_xy = self.INIT_XYZS[0, 0:2] if hasattr(self, 'INIT_XYZS') else np.array([0.0, 0.0])
            target_xy = self.TARGET_POS[0:2]
            D = np.linalg.norm(target_xy - start_xy)
            u = (target_xy - start_xy) / max(D, 1e-4)   # unit vector along line of sight
            u_perp = np.array([-u[1], u[0]])             # perpendicular normal vector

            # Placed squarely in the middle (45% to 55% of the distance)
            frac = np.random.uniform(0.45, 0.55)
            # Small lateral jitter (±0.05m) so it sits right on the drone's direct flight path
            lat_offset = np.random.uniform(-0.05, 0.05)
            xy = start_xy + frac * D * u + lat_offset * u_perp

            # Vertical cylinder obstacle spanning the cage
            radius = 0.10
            length = 2.0
            color = [0.85, 0.25, 0.15, 1.0]
            direction = [0.0, 0.0, 1.0]
            quat = self._getQuatFromZToVec(direction)
            shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=length, physicsClientId=self.CLIENT)
            visual = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=length, rgbaColor=color, physicsClientId=self.CLIENT)
            obs_id = p.createMultiBody(baseMass=0,
                                        baseCollisionShapeIndex=shape,
                                        baseVisualShapeIndex=visual,
                                        basePosition=[xy[0], xy[1], 1.0],
                                        baseOrientation=quat,
                                        physicsClientId=self.CLIENT)
            self.obstacle_ids.append(obs_id)
        else:  # hard: 1 obstacle with slightly wider position randomization
            start_xy = self.INIT_XYZS[0, 0:2] if hasattr(self, 'INIT_XYZS') else np.array([0.0, 0.0])
            target_xy = self.TARGET_POS[0:2]
            D = np.linalg.norm(target_xy - start_xy)
            u = (target_xy - start_xy) / max(D, 1e-4)
            u_perp = np.array([-u[1], u[0]])

            frac = np.random.uniform(0.35, 0.65)
            lat_offset = np.random.uniform(-0.10, 0.10)
            xy = start_xy + frac * D * u + lat_offset * u_perp

            radius = 0.12
            length = 2.0
            color = [0.85, 0.20, 0.10, 1.0]
            direction = [0.0, 0.0, 1.0]
            quat = self._getQuatFromZToVec(direction)
            shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=length, physicsClientId=self.CLIENT)
            visual = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=length, rgbaColor=color, physicsClientId=self.CLIENT)
            obs_id = p.createMultiBody(baseMass=0,
                                        baseCollisionShapeIndex=shape,
                                        baseVisualShapeIndex=visual,
                                        basePosition=[xy[0], xy[1], 1.0],
                                        baseOrientation=quat,
                                        physicsClientId=self.CLIENT)
            self.obstacle_ids.append(obs_id)
            
            # Real flight arena: add surrounding net, PVC frame & mission pads for Hard Mode
            self._addFlightCageNet()
            
        # Clear previous debug visual lines to prevent physics memory buildup
        p.removeAllUserDebugItems(physicsClientId=self.CLIENT)
        
        # Draw a clean 3D green circle (wireframe sphere) using debug lines.
        # Outlines the exact 0.25m target zone and is completely invisible to the depth camera!
        radius = getattr(self, 'TARGET_RADIUS', 0.25)
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
            
        # Ground vertical altitude stalk (shows exact 3D altitude without camera perspective illusion)
        p.addUserDebugLine([self.TARGET_POS[0], self.TARGET_POS[1], 0.0], self.TARGET_POS, lineColorRGB=[0.2, 0.6, 1.0], lineWidth=2, physicsClientId=self.CLIENT)
        
        # Center bullseye crosshair (yellow, marks exact center coordinate)
        p.addUserDebugLine(self.TARGET_POS - np.array([0.06, 0, 0]), self.TARGET_POS + np.array([0.06, 0, 0]), lineColorRGB=[1.0, 0.8, 0.0], lineWidth=3, physicsClientId=self.CLIENT)
        p.addUserDebugLine(self.TARGET_POS - np.array([0, 0.06, 0]), self.TARGET_POS + np.array([0, 0.06, 0]), lineColorRGB=[1.0, 0.8, 0.0], lineWidth=3, physicsClientId=self.CLIENT)
        p.addUserDebugLine(self.TARGET_POS - np.array([0, 0, 0.06]), self.TARGET_POS + np.array([0, 0, 0.06]), lineColorRGB=[1.0, 0.8, 0.0], lineWidth=3, physicsClientId=self.CLIENT)

    @staticmethod
    def _get_or_create_net_texture():
        tex_dir = os.path.join(os.path.dirname(__file__), "..", "assets")
        tex_path = os.path.join(tex_dir, "flight_cage_net.png")
        if not os.path.exists(tex_path):
            try:
                os.makedirs(tex_dir, exist_ok=True)
                size = 512
                # Neutral lab room backdrop behind the net (light grey-beige tone)
                img = np.full((size, size, 3), (210, 215, 218), dtype=np.uint8)
                grid_step = 32
                # Black mesh net cords
                for x in range(0, size, grid_step):
                    cv2.line(img, (x, 0), (x, size), (35, 35, 35), 3, lineType=cv2.LINE_AA)
                for y in range(0, size, grid_step):
                    cv2.line(img, (0, y), (size, y), (35, 35, 35), 3, lineType=cv2.LINE_AA)
                # Knots at grid intersections
                for x in range(0, size, grid_step):
                    for y in range(0, size, grid_step):
                        cv2.circle(img, (x, y), 3, (20, 20, 20), -1, lineType=cv2.LINE_AA)
                cv2.imwrite(tex_path, img)
            except Exception as e:
                print(f"[WARNING] Could not create net texture: {e}")
                return None
        return tex_path

    def _addFlightCageNet(self):
        """
        Builds the 2x2x2 physical flight cage with:
          1. Textured surrounding black mesh net on all 4 walls and ceiling.
          2. White PVC pipe frame: 4 corner posts, 4 mid-span posts, top/mid/bottom rails.
          3. DJI Tello Mission Pads arranged on the floor.
        """
        L = self.BOUND_LIMIT_XY  # 1.0m (2m x 2m x 2m cage)
        H = 2.0                  # 2m height
        
        # Load or create net texture
        tex_path = self._get_or_create_net_texture()
        tex_id = -1
        if tex_path and os.path.exists(tex_path):
            try:
                tex_id = p.loadTexture(tex_path, physicsClientId=self.CLIENT)
            except Exception:
                tex_id = -1

        # 1. Four Net Walls + Ceiling
        wall_configs = [
            # (center, halfExtents)
            ([-L, 0.0, H/2], [0.003, L, H/2]),     # West wall (-X)
            ([ L, 0.0, H/2], [0.003, L, H/2]),     # East wall (+X)
            ([0.0, -L, H/2], [L, 0.003, H/2]),     # South wall (-Y)
            ([0.0,  L, H/2], [L, 0.003, H/2]),     # North wall (+Y)
            ([0.0, 0.0, H],   [L, L, 0.003]),       # Ceiling (+Z)
        ]
        for pos, extents in wall_configs:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=extents, physicsClientId=self.CLIENT)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=extents, rgbaColor=[1.0, 1.0, 1.0, 0.95], physicsClientId=self.CLIENT)
            net_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                                       basePosition=pos, physicsClientId=self.CLIENT)
            if tex_id != -1:
                p.changeVisualShape(net_id, -1, textureUniqueId=tex_id, physicsClientId=self.CLIENT)
            self.net_ids.append(net_id)

        # 2. White PVC Pipe Frame (matching real cage)
        pvc_color = [0.95, 0.95, 0.95, 1.0]  # clean white PVC
        pipe_r = 0.025  # 2.5cm radius

        # Vertical upright posts (4 corners + 4 mid-span posts)
        upright_positions = [
            [-L, -L], [ L, -L], [-L,  L], [ L,  L],   # 4 corners
            [0.0, -L], [0.0,  L], [-L, 0.0], [ L, 0.0], # 4 mid-points
        ]
        for ux, uy in upright_positions:
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=pipe_r, height=H, physicsClientId=self.CLIENT)
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=pipe_r, length=H, rgbaColor=pvc_color, physicsClientId=self.CLIENT)
            post_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                                        basePosition=[ux, uy, H/2], physicsClientId=self.CLIENT)
            self.net_ids.append(post_id)

        # Horizontal rails (Top z=2.0m, Mid-waist z=1.0m, Bottom z=0.03m)
        for z_level in [0.03, 1.0, H]:
            # X-parallel rails (at y = -L and y = +L)
            for ry in [-L, L]:
                col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[L, pipe_r, pipe_r], physicsClientId=self.CLIENT)
                vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[L, pipe_r, pipe_r], rgbaColor=pvc_color, physicsClientId=self.CLIENT)
                rail_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                                            basePosition=[0.0, ry, z_level], physicsClientId=self.CLIENT)
                self.net_ids.append(rail_id)

            # Y-parallel rails (at x = -L and x = +L)
            for rx in [-L, L]:
                col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[pipe_r, L, pipe_r], physicsClientId=self.CLIENT)
                vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[pipe_r, L, pipe_r], rgbaColor=pvc_color, physicsClientId=self.CLIENT)
                rail_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                                            basePosition=[rx, 0.0, z_level], physicsClientId=self.CLIENT)
                self.net_ids.append(rail_id)

        # 3. DJI Tello Mission Pads on the floor (4x4 grid as seen in the photo)
        pad_size = 0.10  # 20cm x 20cm pad
        for px in [-0.60, -0.20, 0.20, 0.60]:
            for py in [-0.60, -0.20, 0.20, 0.60]:
                pad_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[pad_size, pad_size, 0.001],
                                              rgbaColor=[0.12, 0.14, 0.16, 1.0], physicsClientId=self.CLIENT)
                p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=pad_vis,
                                  basePosition=[px, py, 0.001], physicsClientId=self.CLIENT)
                # Pad inner target circle (teal/orange markings matching DJI pads)
                inner_color = [0.15, 0.65, 0.75, 1.0] if (abs(px)+abs(py)) % 0.4 < 0.2 else [0.85, 0.35, 0.20, 1.0]
                circle_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.035, length=0.0015,
                                                 rgbaColor=inner_color, physicsClientId=self.CLIENT)
                p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=circle_vis,
                                  basePosition=[px, py, 0.0015], physicsClientId=self.CLIENT)
        
    def _check_collision(self):
        for obs_id in getattr(self, 'obstacle_ids', []):
            contacts = p.getContactPoints(bodyA=self.DRONE_IDS[0], bodyB=obs_id, physicsClientId=self.CLIENT)
            if contacts is not None and len(contacts) > 0:
                return True
        for net_id in getattr(self, 'net_ids', []):
            contacts = p.getContactPoints(bodyA=self.DRONE_IDS[0], bodyB=net_id, physicsClientId=self.CLIENT)
            if contacts is not None and len(contacts) > 0:
                return True
        # Check ground
        contacts = p.getContactPoints(bodyA=self.DRONE_IDS[0], bodyB=self.PLANE_ID, physicsClientId=self.CLIENT)
        if contacts is not None and len(contacts) > 0:
            return True
        return False
        
    def _get_min_obstacle_distance(self, max_search_dist=0.30):
        """ Returns the closest geometric surface distance between the drone and any obstacle in meters. """
        min_dist = max_search_dist
        for obs_id in getattr(self, 'obstacle_ids', []):
            pts = p.getClosestPoints(bodyA=self.DRONE_IDS[0], bodyB=obs_id, distance=max_search_dist, physicsClientId=self.CLIENT)
            if pts:
                dist = pts[0][8]
                if dist < min_dist:
                    min_dist = dist
        return float(min_dist)

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        distance = np.linalg.norm(self.TARGET_POS - pos)
        
        # 1. Projected Progress (ICARUS dot-product formulation):
        # Rewards displacement along the line-of-sight unit vector toward target.
        # This provides a clean forward gradient into the target without penalizing
        # necessary lateral maneuvers around obstacles.
        delta_p = pos - self.prev_pos
        u_target = (self.TARGET_POS - self.prev_pos) / (self.prev_distance + 1e-6)
        proj_progress = float(np.dot(delta_p, u_target))
        reward = proj_progress * 10.0
        
        # Dedicated vertical altitude progress (ensures swift climb/descent to target altitude):
        dz = abs(self.TARGET_POS[2] - pos[2])
        z_progress = self.prev_dz - dz
        reward += z_progress * 5.0
        self.prev_dz = dz
        
        # 2. Velocity Alignment (ICARUS cosine velocity reward):
        # Rewards flying in the direction of the target, eliminating hovering/stalling near gates.
        vel = state[10:13]
        speed = float(np.linalg.norm(vel))
        min_obs_dist = self._get_min_obstacle_distance(max_search_dist=0.60)
        if speed > 0.02:
            d_target = self.TARGET_POS - pos
            cos_vel_target = float(np.dot(vel, d_target) / (speed * distance + 1e-6))
            # If clear (>0.6m), reward cruising speed; if near obstacle (<0.6m), do not penalize braking/slowing down
            speed_ratio = 1.0 if min_obs_dist < 0.60 else min(speed / self.SPEED_LIMIT, 1.0)
            reward += 0.20 * cos_vel_target * speed_ratio
            
        # 3. Heading alignment: penalize camera misalignment (strictly <= 0.0)
        # cos(heading_err) = 1.0 when perfectly facing target (reward 0.0), -1.0 when reversed (penalty -0.50).
        # This CANNOT be farmed by hovering because standing still earns at most 0.0!
        yaw = state[9]
        dx = self.TARGET_POS[0] - pos[0]
        dy = self.TARGET_POS[1] - pos[1]
        heading_err = np.arctan2(dy, dx) - yaw
        heading_err = (heading_err + np.pi) % (2 * np.pi) - np.pi
        cos_heading = float(np.cos(heading_err))
        reward += (cos_heading - 1.0) * 0.25

        # 4. Penalize excessive spinning so camera remains stable
        yaw_rate = state[15]
        reward -= abs(yaw_rate) * 0.02
        
        # 5. Time Step Cost: creates urgency to reach the target swiftly
        # Every second hovering costs ~1.0 point, making lingering near the target unprofitable
        reward -= 0.04
            
        # 7. Obstacle Proximity & Speed Cushion (active within 0.10m of any obstacle):
        # Updated per user: penalizes being within 0.10m of an obstacle, with extra penalty for rushing at high speed!
        if min_obs_dist < 0.10:
            prox_factor = (0.10 - min_obs_dist) / 0.10  # 0.0 at 0.10m clearance, 1.0 at contact
            reward -= (prox_factor ** 2) * 0.30
            if speed > 0.25:
                reward -= (speed - 0.25) * prox_factor * 0.80
        if min_obs_dist < 0.60:
            prox_factor = (0.60 - min_obs_dist) / 0.60  # 0.0 at 0.6m clearance, 1.0 at contact
            reward -= (prox_factor ** 2) * 0.30
            if speed > 0.25:
                reward -= (speed - 0.25) * prox_factor * 0.80
            
        # 8. Altitude corridor deviation penalty (keeps flight near nominal cruising level)
        dz_excess = max(0.0, abs(pos[2] - self.TARGET_POS[2]) - 0.35)
        if dz_excess > 0.0:
            reward -= dz_excess * 1.5
            
        # 9. Bounded Terminal Events:
        if self._check_collision():
            reward -= getattr(self, 'COLLISION_PENALTY', 8.0)
            
        if abs(pos[0]) > self.BOUND_LIMIT_XY or abs(pos[1]) > self.BOUND_LIMIT_XY or pos[2] > 2.0 or pos[2] < 0.15:
            reward -= getattr(self, 'BOUND_PENALTY', 5.0)
            
        if abs(state[7]) > 1.5 or abs(state[8]) > 1.5:
            reward -= getattr(self, 'BOUND_PENALTY', 5.0)
            
        if distance < self.ARRIVAL_THRESHOLD:
            reward += getattr(self, 'ARRIVAL_BONUS', 15.0)
            
        self.prev_pos = pos.copy()
        self.prev_distance = distance
        return float(reward)
        
    def _computeTerminated(self):
        state = self._getDroneStateVector(0)
        distance = np.linalg.norm(self.TARGET_POS - state[0:3])
        return bool(distance < self.ARRIVAL_THRESHOLD)
        
    def _computeTruncated(self):
        state = self._getDroneStateVector(0)
        if self._check_collision():
            self._truncation_reason = "collision"
            return True
        if abs(state[0]) > self.BOUND_LIMIT_XY or abs(state[1]) > self.BOUND_LIMIT_XY or state[2] > 2.0 or state[2] < 0.15:
            self._truncation_reason = "out_of_bounds"
            return True
        if abs(state[7]) > 1.5 or abs(state[8]) > 1.5:
            self._truncation_reason = "flipped"
            return True
        if (self.step_counter + self.PYB_STEPS_PER_CTRL) / self.PYB_FREQ >= self.EPISODE_LEN_SEC:
            self._truncation_reason = "timeout"
            return True
        self._truncation_reason = None
        return False
        
    def _computeInfo(self):
        state = self._getDroneStateVector(0)
        distance = np.linalg.norm(self.TARGET_POS - state[0:3])
        success = bool(distance < self.ARRIVAL_THRESHOLD)
        return {
            "distance": float(distance),
            "collision": self._check_collision(),
            "success": success,
            "truncation_reason": getattr(self, '_truncation_reason', None),
        }

    def reset(self, seed=None, options=None):
        # Safety margin: target must be at least this far inside the cage boundary
        margin = self.ARRIVAL_THRESHOLD + 0.05
        xy_limit = self.BOUND_LIMIT_XY - margin  # max target XY coord
        edge_margin = 0.15  # how close to the wall the drone spawns

        # --- Pick a random edge and spawn drone near it ---
        # Edge 0: -X wall,  Edge 1: +X wall,  Edge 2: -Y wall,  Edge 3: +Y wall
        edge = np.random.randint(0, 4)

        if edge == 0:      # spawn near -X wall, target in +X half
            spawn_x = -self.BOUND_LIMIT_XY + edge_margin
            spawn_y = np.random.uniform(-0.3, 0.3)
            target_x = np.random.uniform(0.2, xy_limit)
            target_y = np.random.uniform(-xy_limit, xy_limit)
        elif edge == 1:    # spawn near +X wall, target in -X half
            spawn_x = self.BOUND_LIMIT_XY - edge_margin
            spawn_y = np.random.uniform(-0.3, 0.3)
            target_x = np.random.uniform(-xy_limit, -0.2)
            target_y = np.random.uniform(-xy_limit, xy_limit)
        elif edge == 2:    # spawn near -Y wall, target in +Y half
            spawn_x = np.random.uniform(-0.3, 0.3)
            spawn_y = -self.BOUND_LIMIT_XY + edge_margin
            target_x = np.random.uniform(-xy_limit, xy_limit)
            target_y = np.random.uniform(0.2, xy_limit)
        else:              # spawn near +Y wall, target in -Y half
            spawn_x = np.random.uniform(-0.3, 0.3)
            spawn_y = self.BOUND_LIMIT_XY - edge_margin
            target_x = np.random.uniform(-xy_limit, xy_limit)
            target_y = np.random.uniform(-xy_limit, -0.2)

        spawn_z = 1.0  # hovering altitude

        # Set spawn position
        self.INIT_XYZS = np.array([[spawn_x, spawn_y, spawn_z]])

        # Set target position
        if self.DIFFICULTY == 'easy':
            target_z = np.random.uniform(0.8, 1.2)
        elif self.DIFFICULTY == 'medium':
            target_z = np.random.uniform(0.7, 1.3)
        else:
            target_z = np.random.uniform(0.8, 1.3)
        self.TARGET_POS = np.array([target_x, target_y, target_z])

        # Reset estimates
        self.est_pos = np.zeros(2)

        # --- Camera faces forward toward the target ---
        yaw_to_target = np.arctan2(
            self.TARGET_POS[1] - spawn_y,
            self.TARGET_POS[0] - spawn_x,
        )
        if self.DIFFICULTY == 'easy':
            # Easy: small yaw jitter (±30°) so the policy still learns to correct heading
            yaw_jitter = np.random.uniform(-np.radians(30.0), np.radians(30.0))
        else:
            # Hard: tight jitter (±15°) — camera roughly faces the corridor
            yaw_jitter = np.random.uniform(-np.radians(15.0), np.radians(15.0))
        self.INIT_RPYS = np.array([[0.0, 0.0, yaw_to_target + yaw_jitter]])

        obs, info = super().reset(seed=seed, options=options)

        # Initialize prev position, distance, and dz
        state = self._getDroneStateVector(0)
        self.prev_pos = state[0:3].copy()
        self.prev_distance = np.linalg.norm(self.TARGET_POS - state[0:3])
        self.prev_dz = abs(self.TARGET_POS[2] - state[2])
        self.smoothed_action = np.zeros(4)

        # Fill action delay buffer
        delay_steps = np.random.randint(1, 4)
        self.action_buffer = deque(maxlen=delay_steps)
        for _ in range(delay_steps):
            self.action_buffer.append(np.zeros(4))
        self.prev_action = np.zeros(4)

        return obs, info
