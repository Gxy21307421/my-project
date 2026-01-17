import os
from typing import Dict
import numpy as np
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune import register_env

from vmas import make_env
from vmas.simulator.environment import Wrapper

# 场景配置
scenario_name = "balance"

# 场景特定变量
n_agents = 3  # balance 场景默认是 3 个智能体

# 通用变量
continuous_actions = True
max_steps = 100  # 减少最大步数
num_vectorized_envs = 2  # 最小化环境数量
num_workers = 0  # 在本地模式下使用 0
vmas_device = "cuda"  # 重要：改为 "cuda" 以启用 GPU


def env_creator(config: Dict):
    env = make_env(
        scenario=config["scenario_name"],
        num_envs=config["num_envs"],
        device=config["device"],
        continuous_actions=config["continuous_actions"],
        wrapper=Wrapper.RLLIB,  # 关键：使用 RLLIB 包装器
        max_steps=config["max_steps"],
        # 场景特定变量
        **config["scenario_config"],
    )
    return env


# 初始化 Ray
if not ray.is_initialized():
    ray.init(local_mode=True)  # 使用本地模式
    print("Ray init!")

register_env(scenario_name, lambda config: env_creator(config))


def train_balance_cppo_gpu():
    # 设置环境变量以使用 GPU
    os.environ["RLLIB_NUM_GPUS"] = "1"

    tune.run(
        PPOTrainer,
        stop={"training_iteration": 400},  # 训练 400 次迭代
        checkpoint_freq=1,
        checkpoint_at_end=True,
        config={
            "seed": 0,
            "framework": "torch",
            "env": scenario_name,

            "kl_coeff": 0.5,       # 从 0.0 提高，约束策略不要大跳
            "lambda": 0.95,
            "clip_param": 0.1,     # 从 0.2 收紧，减小每次更新幅度
            "vf_loss_coeff": 1.0,
            "entropy_coeff": 0.02,  # 从 0.01 略微增大，保持一定探索

            "train_batch_size": 4000,
            "rollout_fragment_length": 400,
            "sgd_minibatch_size": 64,
            "num_sgd_iter": 5,     # 保持 5，不再用更大的迭代次数

            "num_gpus": 1,         # 使用 1 个 GPU
            "num_workers": num_workers,
            "num_envs_per_worker": num_vectorized_envs,

            "lr": 2e-4,            # 从 5e-4 降低，减弱单步更新强度
            "gamma": 0.99,
            "use_gae": True,
            "batch_mode": "truncate_episodes",

            # === VMAS 环境配置 ===
            "env_config": {
                "device": vmas_device,  # 使用 cuda 设备
                "num_envs": num_vectorized_envs,
                "scenario_name": scenario_name,
                "continuous_actions": continuous_actions,
                "max_steps": max_steps,
                "scenario_config": {
                    "n_agents": n_agents,
                },
            },
        },
    )


if __name__ == "__main__":
    try:
        train_balance_cppo_gpu()
    finally:
        if ray.is_initialized():
            ray.shutdown()