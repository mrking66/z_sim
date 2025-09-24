# train_ppo.py
import os
from datetime import datetime

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.logger import configure

class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        
    def _on_step(self) -> bool:
        if len(self.locals.get("infos", [])) > 0:
            info = self.locals["infos"][0]  # 获取第一个环境的信息
            if "rewards" in info:
                rewards = info["rewards"]
                # 记录每个奖励组件
                for name, value in rewards.items():
                    self.logger.record(f"rewards/{name}", value)
        return True

# 导入你实现的 ZerothEnv（确保 zeroth_env.py 在同一目录并能导入）
from zeroth_env import ZerothEnv

def make_env_fn(xml_path, render_mode=None, seed=0):
    def _init():
        env = ZerothEnv(xml_path=xml_path, render_mode=render_mode, max_steps=20000)
        env.reset(seed=seed)
        return env
    return _init

def main():
    # --------------------------
    # 配置
    # --------------------------
    xml_path = "mechanism/robot_fixed.xml"   # 你的模型文件路径
    
    # 选择是否继续训练
    continue_training = False  # 设置为True来继续训练，False来从头开始
    if continue_training:
        # 使用已有的训练结果目录
        log_dir = "logs/ppo_20250918_111610"
        model_path = os.path.join(log_dir, "checkpoints/ppo_checkpoint_43100000_steps.zip")  # 或使用 "best_model/best_model.zip"
        vec_normalize_path = os.path.join(log_dir, "vec_normalize.pkl")
    else:
        # 创建新的训练目录
        log_dir = os.path.join("logs", "ppo_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(log_dir, exist_ok=True)
        model_path = None
        vec_normalize_path = None

    # 训练参数
    total_timesteps = 100_000_000
    n_envs = 12   # 并行环境个数（CPU越多可以增大）
    eval_freq = 50_000  # 每多少 step 评估一次（整个向量环境的步数）
    n_eval_episodes = 5

    # --------------------------
    # 创建并行环境（训练用不渲染）
    # --------------------------
    # 使用 make_vec_env 简化创建 SubprocVecEnv
    # 注意：render_mode=None 用于训练（no rendering）。
    vec_env = make_vec_env(
        make_env_fn(xml_path, render_mode=None),
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,   # 若本机 Windows 或出现问题，可改为 DummyVecEnv
        seed=0,
    )

    # 对观测和奖励进行归一化（在 RL 中通常很有用）
    if continue_training and vec_normalize_path and os.path.exists(vec_normalize_path):
        # 加载已有的归一化参数
        vec_env = VecNormalize.load(vec_normalize_path, vec_env)
        print(f"Loaded VecNormalize parameters from {vec_normalize_path}")
    else:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # --------------------------
    # 回调：评估、检查点
    # --------------------------
    # 评估环境（单进程，开启渲染为 None，因为评估不需要图形）
    eval_env = DummyVecEnv([make_env_fn(xml_path, render_mode=None, seed=1000)])
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False, clip_obs=10.0)
    # 把训练环境的归一化参数拷贝给 eval_env（保持一致尺度）
    eval_env.obs_rms = vec_env.obs_rms

    # 当达到目标奖励就停止训练（可选）
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=20000.0, verbose=1)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=os.path.join(log_dir, "eval_logs"),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        callback_on_new_best=stop_callback,
    )

    checkpoint_callback = CheckpointCallback(save_freq=100_000 // n_envs, save_path=os.path.join(log_dir, "checkpoints"),
                                             name_prefix="ppo_checkpoint")

    # --------------------------
    # 日志（TensorBoard）
    # --------------------------
    tmp_logger = configure(log_dir, ["stdout", "tensorboard"])

    # --------------------------
    # 创建或加载模型
    # --------------------------
    policy_kwargs = dict(
        # net architecture example
        net_arch=[dict(pi=[256, 256], vf=[256, 256])],
    )

    if continue_training and model_path and os.path.exists(model_path):
        # 加载已有的模型
        model = PPO.load(model_path, env=vec_env, verbose=1)
        # 更新学习率（这是你想要修改的参数）
        model.learning_rate = 3e-4  # 修改学习率
        print(f"Loaded model from {model_path} with new learning rate: {model.learning_rate}")
    else:
        # 创建新模型
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            verbose=1,
            tensorboard_log=log_dir,
            learning_rate=3e-4,  # 新的学习率
            ent_coef=0.001,
            gamma=0.99,
            n_steps=2048 ,
            batch_size=64,
            n_epochs=10,
            policy_kwargs=policy_kwargs,
        )
    
    model.set_logger(tmp_logger)

    # 创建奖励记录器回调
    reward_logger = RewardLoggerCallback()

    # --------------------------
    # 训练
    # --------------------------
    try:
        model.learn(total_timesteps=total_timesteps, 
                   callback=[eval_callback, checkpoint_callback, reward_logger])
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        # 保存最终模型和归一化参数
        model.save(os.path.join(log_dir, "final_model"))
        vec_env.save(os.path.join(log_dir, "vec_normalize.pkl"))
        print(f"Model saved to {log_dir}")

        # 关闭环境
        vec_env.close()
        eval_env.close()

   

if __name__ == "__main__":
    main()
