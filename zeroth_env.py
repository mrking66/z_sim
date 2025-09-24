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

        # 动作空间：10 个电机 [-2, 2]
        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(10,), dtype=np.float32)

        # 观测空间：37 维 (10 pos + 10 vel + 10 frc + 4 quat + 3 gyro)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(37,), dtype=np.float32)

        self.render_mode = render_mode
        self.max_steps = max_steps
        self.step_count = 0
        self.viewer = None

        # 初始化 last_action 为零向量
        self.last_action = np.zeros(self.action_space.shape[0], dtype=np.float32)

    def _get_obs(self):
        # 电机状态
        qpos = [self.data.actuator_length[i] for i in range(10)]
        qvel = [self.data.actuator_velocity[i] for i in range(10)]
        qfrc = [self.data.actuator_force[i] for i in range(10)]

        # 传感器 (IMU)
        quat = self.data.sensor("orientation").data.copy()
        gyro = self.data.sensor("angular-velocity").data.copy()

        obs = np.concatenate([qpos, qvel, qfrc, quat, gyro])
        return obs

    def reset(self, seed=None, options=None):
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        if seed is not None:
            np.random.seed(seed)

        # 重置 last_action
        self.last_action = np.zeros(self.action_space.shape[0], dtype=np.float32)
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self.data.ctrl[:] = action
        #print(len(action))
        mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        obs = self._get_obs()

        # ---- 奖励函数 ----
        # --- 权重系数 (可调参数) ---
        velocity_reward_weight = 0.4        # 降低速度奖励权重
        ctrl_cost_weight = 0.002         # 降低控制成本权重
        alive_reward_weight = 0.4           # 大幅提高存活奖励
        straight_reward_weight = 1        # 降低直线奖励权重  
        upright_reward_weight = 4         # 提高直立奖励权重
        stability_reward_weight = 0.18        # 新增：稳定性奖励
        energy_efficiency_weight = 0.05      # 新增：能量效率奖励
        forward_progress_weight = 1        # 新增：前进距离奖励权重
        symmetry_reward_weight = 0.25         # 新增：对称性奖励权重

        # --- 获取状态 ---
        root_velocity = self.data.qvel[:3]      # 躯干线速度 (vx, vy, vz)
        root_angular_velocity = self.data.qvel[3:6] # 躯干角速度 (wx, wy, wz)
        height = self.data.qpos[2]              # 躯干高度
        forward_distance = -self.data.qpos[1]       # 躯干前进距离 (y 方向)
        root_quat = self.data.sensor("orientation").data  #获取躯干的姿态四元数
        
        #print(f"root_quat: {root_quat}")
        # --- 计算各项奖励/惩罚 ---
#**********************************************************************************************************#
        # 1. 速度追踪奖励: 鼓励以一个目标速度区间前进
        target_velocity_min = -0.16  # 目标速度区间下限 (m/s)
        target_velocity_max = -0.14  # 目标速度区间上限 (m/s)
        forward_velocity = root_velocity[1]
        # 区间内奖励最高，区间外高斯下降
        if target_velocity_min <= forward_velocity <= target_velocity_max:
            velocity_reward = velocity_reward_weight
        else:
            # 距离区间最近边界的距离
            if forward_velocity < target_velocity_min:
                dist = target_velocity_min - forward_velocity
            else:
                dist = forward_velocity - target_velocity_max
            # 高斯型奖励递减
            velocity_reward = velocity_reward_weight * np.exp(-16*dist)
        #print(f"forward_velocity: {forward_velocity:.3f}, velocity_reward: {velocity_reward:.3f}")
        #0~0.43
#**********************************************************************************************************#
        # 2. 控制成本: 惩罚过大的电机指令，鼓励平滑动作
        ctrl_cost = ctrl_cost_weight * np.sum(np.square(action))
        #print(f"ctrl_cost: {ctrl_cost:.6f}")
        #0.0012~0.0038
#**********************************************************************************************************#        
        # 3. 存活奖励: 只要没摔倒就给予奖励
        alive_reward = alive_reward_weight
        #print(f"alive_reward: {alive_reward:.3f}")
        #0.3
#**********************************************************************************************************#
        # 4. 非直线惩罚: 惩罚侧向移动和绕Z轴的旋转
        straight_cost = straight_reward_weight * (np.square(root_velocity[0]) + 0.5*np.square(root_angular_velocity[2]))
        #print(f"straight_cost: {straight_cost:.3f},vx={root_velocity[0]:.3f}, wz={root_angular_velocity[2]:.3f}")
        #0~0.146
#**********************************************************************************************************#
        # 5. 直立奖励: 鼓励躯干保持竖直 (新增)
        # 使用 mujoco 函数将四元数转换为旋转矩阵
        rot_matrix = np.zeros(9, dtype=np.float64)  # 一维数组长度9
        mujoco.mju_quat2Mat(rot_matrix, root_quat)
        rot_matrix = rot_matrix.reshape(3, 3)       # 转为3x3矩阵
        body_z_axis = rot_matrix[:, 2]
        upright_reward = upright_reward_weight * (np.abs(body_z_axis[2])-0.9)
        #print(f"upright_reward: {upright_reward:.3f}, body_z_axis[2]: {body_z_axis[2]:.3f}")
        #-1.1~0.45
#**********************************************************************************************************#
        # 6. 新增：稳定性奖励 - 奖励较小的角速度变化
        stability_reward = stability_reward_weight * np.exp(-0.5 * np.sum(np.square(root_angular_velocity)))
        #print(f"stability_reward: {stability_reward:.3f}")
        #0~0.14
#**********************************************************************************************************#
        # 7. 新增：能量效率奖励 - 奖励动作的平滑性
        if hasattr(self, 'last_action'):
            action_smoothness = np.sum(np.square(action - self.last_action))
            energy_efficiency_reward = energy_efficiency_weight * np.exp(-action_smoothness)
        else:
            energy_efficiency_reward = 0
        self.last_action = action.copy()
        #print(f"energy_efficiency_reward: {energy_efficiency_reward:.3f}")
        #0

#**********************************************************************************************************#
        # --- #8 机身高度奖励与摔倒惩罚 (高斯形式) ---
        # 定义理想高度范围
        min_safe_height = 0.25
        max_safe_height = 0.33
        target_height = (min_safe_height + max_safe_height) / 2.0 # 目标中心高度 0.28
        # 定义权重和惩罚系数
        height_reward_weight = 1.0  # 在理想范围内的最大奖励
        penalty_steepness = 40.0    # 惩罚区域的陡峭程度
        fall_penalty = -30          # 摔倒的巨大惩罚
        if height < 0.18:
            # 1. 摔倒区域: 给予巨大的固定惩罚
            height_reward_penalty = fall_penalty
        elif height < min_safe_height:
            # 2. 警告区域 (0.18 ~ 0.25): 给予指数增长的惩罚
            # 离 min_safe_height 越远，惩罚越大
            deviation = min_safe_height - height
            height_reward_penalty = - (np.exp(penalty_steepness * deviation) - 1)
        else:
            # 3. 安全/奖励区域 (>= 0.25): 给予平顶高斯奖励
            # 在 [0.25, 0.31] 范围内奖励接近最大值，之外平滑下降
            # 使用两个 sigmoid 函数构造平顶高斯
            upper_bound = 1 / (1 + np.exp(-20 * (height - min_safe_height)))
            lower_bound = 1 / (1 + np.exp(20 * (height - max_safe_height)))
            in_range_factor = upper_bound * lower_bound
            height_reward_penalty = height_reward_weight * in_range_factor
        #print(f"height: {height:.3f}, height_reward_penalty: {height_reward_penalty:.3f}")
#**********************************************************************************************************#
        # 9. (新增) 前进距离奖励: 走得越远，奖励越高
        # 仅在高度安全且竖直时给予奖励，否则为0，防止摔倒时奖励异常
        upright_threshold = 0.92  # body_z_axis[2] > 0.92 认为竖直
        # --- 静止惩罚相关变量 ---
        if not hasattr(self, 'last_check_step'):
            self.last_check_step = 0
        if not hasattr(self, 'last_check_distance'):
            self.last_check_distance = forward_distance
        still_penalty_value = -0.4  # 惩罚值，可调
        check_interval = 200        # 检查步数间隔
        min_delta = 0.03           # 最小增量阈值

        # --- 前进奖励 ---
        if (height >= min_safe_height) and (body_z_axis[2] > upright_threshold) and forward_distance > 0.03:
            # 指数型前进奖励，越远奖励增长更快
            forward_progress_reward = forward_progress_weight * np.exp(3*forward_distance)-1
            # 每隔100步检查一次静止惩罚
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
        action[0] 控制 right_hip_pitch (右髋关节俯仰)
        action[1] 控制 right_hip_yaw (右髋关节偏航)
        action[2] 控制 right_hip_roll (右髋关节翻滚)
        action[3] 控制 right_knee_pitch (右膝关节俯仰)
        action[4] 控制 right_ankle_pitch (右踝关节俯仰)
        action[5] 控制 left_hip_pitch (左髋关节俯仰)
        action[6] 控制 left_hip_yaw (左髋关节偏航)
        action[7] 控制 left_hip_roll (左髋关节翻滚)
        action[8] 控制 left_knee_pitch (左膝关节俯仰)
        action[9] 控制 left_ankle_pitch (左踝关节俯仰)
        '''
        # 10. (新增) 对称性奖励: 鼓励左右腿动作对称
        # 假设动作数组中，前5个是右腿，后5个是左腿
        right_leg_actions = action[:5]
        left_leg_actions = action[5:]
        # 我们关心髋关节和膝关节的对称性，通常是主要驱动关节
        # 假设 index 0, 1, 2 是髋关节, index 3, 4 是膝关节和踝关节
        symmetry_diff=0
        for i in range(1,5):
            symmetry_diff += np.abs(left_leg_actions[i] + right_leg_actions[i])
        symmetry_reward = symmetry_reward_weight * np.exp(-symmetry_diff)
#**********************************************************************************************************#
         
         # --- 汇总总奖励 ---
        reward = (velocity_reward - ctrl_cost + alive_reward - straight_cost + 
                upright_reward + stability_reward + energy_efficiency_reward + 
                height_reward_penalty+ forward_progress_reward + symmetry_reward)
        #print(f"velocity: {velocity_reward:.3f}, ctrl_cost: {ctrl_cost:.3f}, alive: {alive_reward:.3f}, straight: {straight_cost:.3f}, upright: {upright_reward:.3f}, stability: {stability_reward:.3f}, energy: {energy_efficiency_reward:.3f}, height: {height_reward_penalty:.3f}, forward: {forward_progress_reward:.3f}, sym: {symmetry_reward:.3f}")
        # 将各分项奖励存入info字典，以便在TensorBoard中查看
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
        
        # ---- done 条件 ----
        terminated = height < 0.18  # 摔倒
        truncated = self.step_count >= self.max_steps  # 时间到

        # ---- 渲染 ----
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
        '''
        # 打印 qpos[0], qpos[1]
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
