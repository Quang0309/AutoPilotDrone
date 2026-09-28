import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import argparse
import time
from datetime import datetime
import numpy as np
import torch
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.envs.NavigateAviary import NavigateAviary
from gym_pybullet_drones.utils.NavigateExtractor import NavigateExtractor
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.utils import str2bool

DEFAULT_GUI = False
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_TIMESTEPS = 3000000
DEFAULT_N_ENVS = 4

def run(gui=DEFAULT_GUI, record_video=DEFAULT_RECORD_VIDEO, output_folder=DEFAULT_OUTPUT_FOLDER, timesteps=DEFAULT_TIMESTEPS, n_envs=DEFAULT_N_ENVS, resume_model=None, difficulty='easy', lr=3e-4):
    
    filename = os.path.join(output_folder, 'navigate', datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename+'/')
        
    print(f"[INFO] Initializing {n_envs} parallel environments with difficulty: {difficulty}...")
    
    train_env = make_vec_env(NavigateAviary,
                             env_kwargs=dict(obs=ObservationType.ALL, act=ActionType.VEL, gui=False, difficulty=difficulty),
                             n_envs=n_envs,
                             seed=0)
    train_env = VecFrameStack(train_env, n_stack=4, channels_order="first")
                             
    eval_env = NavigateAviary(obs=ObservationType.ALL, act=ActionType.VEL, gui=gui, record=record_video, difficulty=difficulty)
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecFrameStack(eval_env, n_stack=4, channels_order="first")
    
    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)
    
    # Custom policy kwargs for the slim MultiInputPolicy
    # NOTE: No activation_fn override — uses SB3 default Tanh, which matches the working MlpPolicy
    policy_kwargs = dict(
        features_extractor_class=NavigateExtractor,
        features_extractor_kwargs=dict(cnn_output_dim=32),
        share_features_extractor=True,
        net_arch=[64, 64]
    )
    
    if resume_model is not None and os.path.exists(resume_model):
        print(f"[INFO] Resuming training from {resume_model} with lr={lr}")
        model = PPO.load(resume_model, env=train_env, tensorboard_log=filename+'/tb/', device="mps", custom_objects={"learning_rate": lr})
    else:
        model = PPO("MultiInputPolicy",
                    train_env,
                    policy_kwargs=policy_kwargs,
                    tensorboard_log=filename+'/tb/',
                    device="mps",
                    verbose=1,
                    learning_rate=lr,
                    n_steps=2048,
                    batch_size=1024,
                    n_epochs=10
                    )
    
    # [CRITICAL FIX] Freeze CNN parameters during Easy Mode
    # Zero-init only silences the CNN at step 0. Without freezing, gradients
    # push the weights away from zero, causing the CNN to emit random noise
    # that drowns out the compass signal — exactly the same failure mode.
    if difficulty == 'easy':
        frozen_count = 0
        for name, param in model.policy.named_parameters():
            if 'cnn' in name:
                param.requires_grad = False
                frozen_count += param.numel()
        print(f"[INFO] Frozen {frozen_count:,} CNN parameters for Easy Mode")
    else:
        # Hard Mode: explicitly unfreeze CNN so it can learn from real depth
        unfrozen_count = 0
        for name, param in model.policy.named_parameters():
            if 'cnn' in name:
                param.requires_grad = True
                unfrozen_count += param.numel()
        print(f"[INFO] Unfrozen {unfrozen_count:,} CNN parameters for Hard Mode")
                
    # callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=150.0, verbose=1)
    
    eval_callback = EvalCallback(eval_env,
                                 verbose=1,
                                 best_model_save_path=filename+'/',
                                 log_path=filename+'/',
                                 eval_freq=int(5000 / n_envs),
                                 n_eval_episodes=10,
                                 deterministic=True,
                                 render=False)
                                 
    print(f"[INFO] Starting training for {timesteps} timesteps...")
    start_time = time.time()
    
    model.learn(total_timesteps=timesteps,
                callback=eval_callback,
                log_interval=10)
                
    end_time = time.time()
    print(f"[INFO] Training finished in {(end_time - start_time)/60:.2f} minutes.")
    
    model.save(filename+'/final_model.zip')
    print(f"[INFO] Model saved to {filename}")
    
    # Evaluation
    print("[INFO] Evaluating the trained model...")
    best_model = PPO.load(filename+'/best_model.zip')
    mean_reward, std_reward = evaluate_policy(best_model, eval_env, n_eval_episodes=10)
    print(f"[INFO] Mean Reward: {mean_reward} +/- {std_reward}")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train NavigateAviary')
    parser.add_argument('--gui', default=DEFAULT_GUI, type=str2bool, help='Whether to use PyBullet GUI (default: False)', metavar='')
    parser.add_argument('--record_video', default=DEFAULT_RECORD_VIDEO, type=str2bool, help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str, help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--timesteps', default=DEFAULT_TIMESTEPS, type=int, help='Total timesteps to train (default: 3M)', metavar='')
    parser.add_argument('--n_envs', default=DEFAULT_N_ENVS, type=int, help='Number of parallel envs (default: 4)', metavar='')
    parser.add_argument('--resume_model', default=None, type=str, help='Path to a .zip model to resume training from', metavar='')
    parser.add_argument('--difficulty', default='easy', type=str, choices=['easy', 'medium', 'hard'], help='Curriculum difficulty level (easy, medium, or hard)', metavar='')
    parser.add_argument('--lr', default=3e-4, type=float, help='Learning rate (default: 3e-4)', metavar='')
    
    ARGS = parser.parse_args()
    run(**vars(ARGS))
