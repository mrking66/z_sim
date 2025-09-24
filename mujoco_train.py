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
            info = self.locals["infos"][0]  # Get info from the first environment
            if "rewards" in info:
                rewards = info["rewards"]
                # Log each reward component
                for name, value in rewards.items():
                    self.logger.record(f"rewards/{name}", value)
        return True

# Import your implemented ZerothEnv (ensure zeroth_env.py is in the same directory and can be imported)
from mujoco_env import ZerothEnv

def make_env_fn(xml_path, render_mode=None, seed=0):
    def _init():
        env = ZerothEnv(xml_path=xml_path, render_mode=render_mode, max_steps=20000)
        env.reset(seed=seed)
        return env
    return _init

def main():
    # --------------------------
    # Configuration
    # --------------------------
    xml_path = "mechanism/robot_fixed.xml"   # Your model file path
    
    # Choose whether to continue training
    continue_training = False  # Set to True to continue training, False to start from scratch
    if continue_training:
        # Use existing training results directory
        log_dir = "logs/ppo_20250918_111610"
        model_path = os.path.join(log_dir, "checkpoints/ppo_checkpoint_43100000_steps.zip")  # Or use "best_model/best_model.zip"
        vec_normalize_path = os.path.join(log_dir, "vec_normalize.pkl")
    else:
        # Create new training directory
        log_dir = os.path.join("logs", "ppo_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(log_dir, exist_ok=True)
        model_path = None
        vec_normalize_path = None

    # Training parameters
    total_timesteps = 100_000_000
    n_envs = 12   # Number of parallel environments (can increase with more CPUs)
    eval_freq = 50_000  # Evaluate every how many steps (steps across the vector environment)
    n_eval_episodes = 5

    # --------------------------
    # Create parallel environments (for training, no rendering)
    # --------------------------
    # Use make_vec_env to simplify creating SubprocVecEnv
    # Note: render_mode=None for training (no rendering).
    vec_env = make_vec_env(
        make_env_fn(xml_path, render_mode=None),
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,   # If on Windows or having issues, can change to DummyVecEnv
        seed=0,
    )

    # Normalize observations and rewards (usually helpful in RL)
    if continue_training and vec_normalize_path and os.path.exists(vec_normalize_path):
        # Load existing normalization parameters
        vec_env = VecNormalize.load(vec_normalize_path, vec_env)
        print(f"Loaded VecNormalize parameters from {vec_normalize_path}")
    else:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # --------------------------
    # Callbacks: Evaluation, Checkpoints
    # --------------------------
    # Evaluation environment (single process, render_mode=None as evaluation doesn't need graphics)
    eval_env = DummyVecEnv([make_env_fn(xml_path, render_mode=None, seed=1000)])
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False, clip_obs=10.0)
    # Copy normalization parameters from training environment to eval_env (maintain consistent scale)
    eval_env.obs_rms = vec_env.obs_rms

    # Stop training when target reward is reached (optional)
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
    # Logging (TensorBoard)
    # --------------------------
    tmp_logger = configure(log_dir, ["stdout", "tensorboard"])

    # --------------------------
    # Create or load model
    # --------------------------
    policy_kwargs = dict(
        # net architecture example
        net_arch=[dict(pi=[256, 256], vf=[256, 256])],
    )

    if continue_training and model_path and os.path.exists(model_path):
        # Load existing model
        model = PPO.load(model_path, env=vec_env, verbose=1)
        # Update learning rate (parameter you want to modify)
        model.learning_rate = 3e-4  # Modify learning rate
        print(f"Loaded model from {model_path} with new learning rate: {model.learning_rate}")
    else:
        # Create new model
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            verbose=1,
            tensorboard_log=log_dir,
            learning_rate=3e-4,  # New learning rate
            ent_coef=0.001,
            gamma=0.99,
            n_steps=2048 ,
            batch_size=64,
            n_epochs=10,
            policy_kwargs=policy_kwargs,
        )
    
    model.set_logger(tmp_logger)

    # Create reward logger callback
    reward_logger = RewardLoggerCallback()

    # --------------------------
    # Training
    # --------------------------
    try:
        model.learn(total_timesteps=total_timesteps, 
                   callback=[eval_callback, checkpoint_callback, reward_logger])
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        # Save final model and normalization parameters
        model.save(os.path.join(log_dir, "final_model"))
        vec_env.save(os.path.join(log_dir, "vec_normalize.pkl"))
        print(f"Model saved to {log_dir}")

        # Close environments
        vec_env.close()
        eval_env.close()

   

if __name__ == "__main__":
    main()