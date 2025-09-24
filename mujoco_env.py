import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer

class ZerothEnv(gym.Env):
    metadata = {"render_modes": ["human", "none"]}

    def __init__(self, xml_path="mechanism/robot_fixed.xml", render_mode=None, max_steps=20000):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Action space: 10 motors [-2, 2]
        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(10,), dtype=np.float32)

        # Observation space: 37 dimensions (10 pos + 10 vel + 10 frc + 4 quat + 3 gyro)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(37,), dtype=np.float32)

        self.render_mode = render_mode
        self.max_steps = max_steps
        self.step_count = 0
        self.viewer = None

        # Initialize last_action as zero vector
        self.last_action = np.zeros(self.action_space.shape[0], dtype=np.float32)

    def _get_obs(self):
        # Motor states
        qpos = [self.data.actuator_length[i] for i in range(10)]
        qvel = [self.data.actuator_velocity[i] for i in range(10)]
        qfrc = [self.data.actuator_force[i] for i in range(10)]

        # Sensors (IMU)
        quat = self.data.sensor("orientation").data.copy()
        gyro = self.data.sensor("angular-velocity").data.copy()

        obs = np.concatenate([qpos, qvel, qfrc, quat, gyro])
        return obs

    def reset(self, seed=None, options=None):
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        if seed is not None:
            np.random.seed(seed)

        # Reset last_action
        self.last_action = np.zeros(self.action_space.shape[0], dtype=np.float32)
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self.data.ctrl[:] = action
        #print(len(action))
        mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        obs = self._get_obs()

        # ---- Reward Function ----
        # --- Weight Coefficients (Adjustable Parameters) ---
        velocity_reward_weight = 0.4        # Decrease velocity reward weight
        ctrl_cost_weight = 0.002           # Decrease control cost weight
        alive_reward_weight = 0.4          # Significantly increase survival reward
        straight_reward_weight = 1         # Decrease straight line reward weight
        upright_reward_weight = 4          # Increase upright reward weight
        stability_reward_weight = 0.18     # New: stability reward
        energy_efficiency_weight = 0.05    # New: energy efficiency reward
        forward_progress_weight = 1        # New: forward progress reward weight
        symmetry_reward_weight = 0.25      # New: symmetry reward weight

        # --- Get States ---
        root_velocity = self.data.qvel[:3]           # Trunk linear velocity (vx, vy, vz)
        root_angular_velocity = self.data.qvel[3:6]  # Trunk angular velocity (wx, wy, wz)
        height = self.data.qpos[2]                   # Trunk height
        forward_distance = -self.data.qpos[1]        # Forward distance (y direction)
        root_quat = self.data.sensor("orientation").data  # Get trunk orientation quaternion
        
        #print(f"root_quat: {root_quat}")
        # --- Calculate various rewards/penalties ---
#**********************************************************************************************************#
        # 1. Velocity tracking reward: Encourage moving within a target velocity range
        target_velocity_min = -0.16  # Target velocity range lower bound (m/s)
        target_velocity_max = -0.14  # Target velocity range upper bound (m/s)
        forward_velocity = root_velocity[1]
        # Maximum reward within the range, Gaussian decay outside the range
        if target_velocity_min <= forward_velocity <= target_velocity_max:
            velocity_reward = velocity_reward_weight
        else:
            # Distance to the nearest boundary of the range
            if forward_velocity < target_velocity_min:
                dist = target_velocity_min - forward_velocity
            else:
                dist = forward_velocity - target_velocity_max
            # Gaussian-shaped reward decay
            velocity_reward = velocity_reward_weight * np.exp(-16*dist)
        #print(f"forward_velocity: {forward_velocity:.3f}, velocity_reward: {velocity_reward:.3f}")
        #0~0.43
#**********************************************************************************************************#
        # 2. Control cost: Penalize large motor commands, encourage smooth actions
        ctrl_cost = ctrl_cost_weight * np.sum(np.square(action))
        #print(f"ctrl_cost: {ctrl_cost:.6f}")
        #0.0012~0.0038
#**********************************************************************************************************#        
        # 3. Survival reward: Reward for not falling down
        alive_reward = alive_reward_weight
        #print(f"alive_reward: {alive_reward:.3f}")
        #0.3
#**********************************************************************************************************#
        # 4. Non-straight penalty: Penalize lateral movement and rotation around Z-axis
        straight_cost = straight_reward_weight * (np.square(root_velocity[0]) + 0.5*np.square(root_angular_velocity[2]))
        #print(f"straight_cost: {straight_cost:.3f},vx={root_velocity[0]:.3f}, wz={root_angular_velocity[2]:.3f}")
        #0~0.146
#**********************************************************************************************************#
        # 5. Upright reward: Encourage trunk to stay upright (new)
        # Use mujoco function to convert quaternion to rotation matrix
        rot_matrix = np.zeros(9, dtype=np.float64)  # 1D array length 9
        mujoco.mju_quat2Mat(rot_matrix, root_quat)
        rot_matrix = rot_matrix.reshape(3, 3)       # Reshape to 3x3 matrix
        body_z_axis = rot_matrix[:, 2]
        upright_reward = upright_reward_weight * (np.abs(body_z_axis[2])-0.9)
        #print(f"upright_reward: {upright_reward:.3f}, body_z_axis[2]: {body_z_axis[2]:.3f}")
        #-1.1~0.45
#**********************************************************************************************************#
        # 6. New: Stability reward - Reward small angular velocity changes
        stability_reward = stability_reward_weight * np.exp(-0.5 * np.sum(np.square(root_angular_velocity)))
        #print(f"stability_reward: {stability_reward:.3f}")
        #0~0.14
#**********************************************************************************************************#
        # 7. New: Energy efficiency reward - Reward action smoothness
        if hasattr(self, 'last_action'):
            action_smoothness = np.sum(np.square(action - self.last_action))
            energy_efficiency_reward = energy_efficiency_weight * np.exp(-action_smoothness)
        else:
            energy_efficiency_reward = 0
        self.last_action = action.copy()
        #print(f"energy_efficiency_reward: {energy_efficiency_reward:.3f}")
        #0

#**********************************************************************************************************#
        # --- #8 Body height reward and fall penalty (Gaussian form) ---
        # Define ideal height range
        min_safe_height = 0.25
        max_safe_height = 0.33
        target_height = (min_safe_height + max_safe_height) / 2.0 # Target center height 0.28
        # Define weights and penalty coefficients
        height_reward_weight = 1.0  # Maximum reward within ideal range
        penalty_steepness = 40.0    # Steepness of penalty area
        fall_penalty = -30          # Large penalty for falling
        if height < 0.18:
            # 1. Fall area: Give large fixed penalty
            height_reward_penalty = fall_penalty
        elif height < min_safe_height:
            # 2. Warning area (0.18 ~ 0.25): Give exponentially increasing penalty
            # The farther from min_safe_height, the greater the penalty
            deviation = min_safe_height - height
            height_reward_penalty = - (np.exp(penalty_steepness * deviation) - 1)
        else:
            # 3. Safe/reward area (>= 0.25): Give flat-top Gaussian reward
            # Reward is near maximum in [0.25, 0.31] range, smoothly decreasing outside
            # Use two sigmoid functions to construct flat-top Gaussian
            upper_bound = 1 / (1 + np.exp(-20 * (height - min_safe_height)))
            lower_bound = 1 / (1 + np.exp(20 * (height - max_safe_height)))
            in_range_factor = upper_bound * lower_bound
            height_reward_penalty = height_reward_weight * in_range_factor
        #print(f"height: {height:.3f}, height_reward_penalty: {height_reward_penalty:.3f}")
#**********************************************************************************************************#
        # 9. (New) Forward progress reward: The farther you go, the higher the reward
        # Only reward when height is safe and upright, otherwise 0, prevent abnormal rewards when falling
        upright_threshold = 0.92  # Consider upright when body_z_axis[2] > 0.92
        # --- Stationary penalty related variables ---
        if not hasattr(self, 'last_check_step'):
            self.last_check_step = 0
        if not hasattr(self, 'last_check_distance'):
            self.last_check_distance = forward_distance
        still_penalty_value = -0.4  # Adjustable penalty value
        check_interval = 200        # Check step interval
        min_delta = 0.03           # Minimum increment threshold

        # --- Forward reward ---
        if (height >= min_safe_height) and (body_z_axis[2] > upright_threshold) and forward_distance > 0.03:
            # Exponential forward reward, faster growth the farther you go
            forward_progress_reward = forward_progress_weight * np.exp(3*forward_distance)-1
            # Check stationary penalty every 100 steps
            if (self.step_count - self.last_check_step) >= check_interval:
                delta = forward_distance - self.last_check_distance
                if delta < min_delta:
                    forward_progress_reward += still_penalty_value
                self.last_check_step = self.step_count
                self.last_check_distance = forward_distance
        else:
            forward_progress_reward = -0.5
        #print(f"forward_distance: {forward_distance:.3f}, forward_progress_reward: {forward_progress_reward:.3f}")
#**********************************************************************************************************#
        '''
        action[0] controls right_hip_pitch (right hip joint pitch)
        action[1] controls right_hip_yaw (right hip joint yaw)
        action[2] controls right_hip_roll (right hip joint roll)
        action[3] controls right_knee_pitch (right knee joint pitch)
        action[4] controls right_ankle_pitch (right ankle joint pitch)
        action[5] controls left_hip_pitch (left hip joint pitch)
        action[6] controls left_hip_yaw (left hip joint yaw)
        action[7] controls left_hip_roll (left hip joint roll)
        action[8] controls left_knee_pitch (left knee joint pitch)
        action[9] controls left_ankle_pitch (left ankle joint pitch)
        '''
        # 10. (New) Symmetry reward: Encourage symmetrical left-right leg movements
        # Assume in action array, first 5 are right leg, last 5 are left leg
        right_leg_actions = action[:5]
        left_leg_actions = action[5:]
        # We care about symmetry of hip and knee joints, usually the main driving joints
        # Assume index 0, 1, 2 are hip joints, index 3, 4 are knee and ankle joints
        symmetry_diff=0
        for i in range(1,5):
            symmetry_diff += np.abs(left_leg_actions[i] + right_leg_actions[i])
        symmetry_reward = symmetry_reward_weight * np.exp(-symmetry_diff)
#**********************************************************************************************************#
         
         # --- Summarize total reward ---
        reward = (velocity_reward - ctrl_cost + alive_reward - straight_cost + 
                upright_reward + stability_reward + energy_efficiency_reward + 
                height_reward_penalty+ forward_progress_reward + symmetry_reward)
        #print(f"velocity: {velocity_reward:.3f}, ctrl_cost: {ctrl_cost:.3f}, alive: {alive_reward:.3f}, straight: {straight_cost:.3f}, upright: {upright_reward:.3f}, stability: {stability_reward:.3f}, energy: {energy_efficiency_reward:.3f}, height: {height_reward_penalty:.3f}, forward: {forward_progress_reward:.3f}, sym: {symmetry_reward:.3f}")
        # Store each component reward in info dictionary for viewing in TensorBoard
        info = {
            "rewards": {
                "velocity": velocity_reward,
                "ctrl_cost": -ctrl_cost,
                "alive": alive_reward,
                "straight": -straight_cost,
                "upright": upright_reward,
                "stability": stability_reward,
                "energy": energy_efficiency_reward,
                "height": height_reward_penalty,
                "forward": forward_progress_reward,
                "symmetry": symmetry_reward,
                "total_reward": reward
            }
        }
        
        # ---- done conditions ----
        terminated = height < 0.18  # Fallen
        truncated = self.step_count >= self.max_steps  # Timeout

        # ---- Rendering ----
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
        '''
        # Print qpos[0], qpos[1]
        x = self.data.qpos[0]
        y = self.data.qpos[1]
        z = self.data.qpos[2]
        print(f"x={x:.3f}, y={y:.3f}, z={z:.3f},vx={root_velocity[0]:.3f}, vy={root_velocity[1]:.3f}, vz={root_velocity[2]:.3f}")
        '''
        return obs, reward, terminated, truncated, info

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None



if __name__ == "__main__":
    env = ZerothEnv(xml_path="mechanism/robot_fixed.xml", render_mode="human", max_steps=10000)
    obs, info = env.reset()
    #print(f"env.data.actuator_length:{env.data.actuator_length}")
    #print(f"env.data.actuator_velocity:{env.data.actuator_velocity}")
    #print(f"env.data.actuator_force:{env.data.actuator_force}")
    
    for _ in range(10000):
        action = env.action_space.sample()
        action[0]=2
        action[5]=2
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()