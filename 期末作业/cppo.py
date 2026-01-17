import os
from typing import Dict

import numpy as np
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune import register_env

import torch
import torch.nn as nn

from vmas import make_env
from vmas.simulator.environment import Wrapper

# 场景配置
scenario_name = "balance"

# 场景特定变量
n_agents = 3  # balance 场景默认是 3 个智能体

# 通用变量 —— 与 ippo 保持一致
continuous_actions = True
max_steps = 100          # 每个 episode 最大步数
num_vectorized_envs = 2  # 每个 worker 上的并行环境数量
num_workers = 0          # 在本地模式下使用 0
vmas_device = "cuda"     # 如需在 CPU 上跑可改为 "cpu"


def env_creator(config: Dict):
    """使用 VMAS 官方提供的 RLlib 包装器创建环境"""
    env = make_env(
        scenario=config["scenario_name"],
        num_envs=config["num_envs"],
        device=config["device"],
        continuous_actions=config["continuous_actions"],
        wrapper=Wrapper.RLLIB,
        max_steps=config["max_steps"],
        # 场景特定变量
        **config["scenario_config"],
    )
    return env


# 初始化 Ray（本地调试模式）
if not ray.is_initialized():
    ray.init(local_mode=True)
    print("Ray init!")

register_env(scenario_name, lambda config: env_creator(config))



class CentralizedActorCriticModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        if hasattr(obs_space, "shape") and obs_space.shape is not None:
            self.obs_dim = int(np.prod(obs_space.shape))
        elif isinstance(obs_space, int):
            self.obs_dim = obs_space
        else:
            self.obs_dim = num_outputs
        self.action_dim = num_outputs

        print(
            f"CentralizedActorCriticModel init: "
            f"obs_dim={self.obs_dim}, action_dim={self.action_dim}, num_outputs={num_outputs}"
        )
        hidden_size = 256
        self.policy_net = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.action_dim),
        )
        self.value_net = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self._value_out = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        if isinstance(obs, torch.Tensor):
            obs_t = obs
        else:
            obs_np = np.asarray(obs, dtype=np.float32)
            obs_t = torch.as_tensor(obs_np, dtype=torch.float32)
        if obs_t.dim() > 2:
            obs_t = obs_t.view(obs_t.size(0), -1)

        logits = self.policy_net(obs_t)
        value = self.value_net(obs_t).squeeze(-1)

        self._value_out = value
        return logits, state

    def value_function(self):

        if self._value_out is None:
            return torch.zeros(1)
        return self._value_out


# 注册 CPPO 模型到 RLlib
ModelCatalog.register_custom_model("centralized_model", CentralizedActorCriticModel)
# ==================== 模型定义结束 ====================


def train_balance_cppo_gpu():
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
            "kl_coeff": 0.5,       # 约束策略不要大跳
            "lambda": 0.95,
            "clip_param": 0.1,     # 收紧 clip，减小每次更新幅度
            "vf_loss_coeff": 1.0,
            "entropy_coeff": 0.02,  # 保持一定探索
            "train_batch_size": 4000,
            "rollout_fragment_length": 400,
            "sgd_minibatch_size": 64,
            "num_sgd_iter": 5,
            "num_gpus": 1,         # 使用 1 个 GPU（在 local_mode 下只影响设备）
            "num_workers": num_workers,
            "num_envs_per_worker": num_vectorized_envs,
            "lr": 2e-4,            # 降低学习率，避免策略崩盘
            "gamma": 0.99,
            "use_gae": True,
            "batch_mode": "truncate_episodes",
            "model": {
                "custom_model": "centralized_model",
                "custom_model_config": {
                    "env_config": {
                        "device": vmas_device,
                        "num_envs": num_vectorized_envs,
                        "scenario_name": scenario_name,
                        "continuous_actions": continuous_actions,
                        "max_steps": max_steps,
                        "scenario_config": {
                            "n_agents": n_agents,
                        },
                    }
                },
            },
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