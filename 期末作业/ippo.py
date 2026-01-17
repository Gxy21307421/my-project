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

# 通用变量 —— 与 mappo 保持一致
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


# ==================== IPPO 模型定义 ====================
class IndependentPPOModel(TorchModelV2, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        env_config = model_config.get("custom_model_config", {}).get("env_config", {})
        self.n_agents = env_config.get("scenario_config", {}).get("n_agents", 1)

        if hasattr(obs_space, "shape") and obs_space.shape is not None:
            self.local_obs_dim = int(np.prod(obs_space.shape))
        elif isinstance(obs_space, int):
            self.local_obs_dim = obs_space
        else:
            self.local_obs_dim = num_outputs
        self.action_dim = num_outputs

        print(
            f"IPPO模型初始化: n_agents={self.n_agents}, "
            f"local_obs_dim={self.local_obs_dim}, "
            f"action_dim={self.action_dim}"
        )

        actor_hidden_size = 256
        self.actor_net = nn.Sequential(
            nn.Linear(self.local_obs_dim, actor_hidden_size),
            nn.ReLU(),
            nn.Linear(actor_hidden_size, actor_hidden_size),
            nn.ReLU(),
            nn.Linear(actor_hidden_size, self.action_dim),
        )
        critic_hidden_size = 256
        self.critic_net = nn.Sequential(
            nn.Linear(self.local_obs_dim, critic_hidden_size),
            nn.ReLU(),
            nn.Linear(critic_hidden_size, critic_hidden_size),
            nn.ReLU(),
            nn.Linear(critic_hidden_size, 1),
        )

        self._value_out = None
        self._local_obs = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0.0)

    def _extract_local_obs(self, obs):
        # Tensor: 直接视为本地观测
        if isinstance(obs, torch.Tensor):
            obs_t = obs.float()
            if obs_t.dim() == 1:
                obs_t = obs_t.unsqueeze(0)
            return obs_t

        # List / tuple: 简化处理，取第一个元素当作本地观测
        if isinstance(obs, (list, tuple)):
            if len(obs) > 0:
                local_obs = obs[0]
                if isinstance(local_obs, torch.Tensor):
                    local_obs_t = local_obs.float()
                else:
                    local_obs_t = torch.as_tensor(local_obs, dtype=torch.float32)
                if local_obs_t.dim() == 1:
                    local_obs_t = local_obs_t.unsqueeze(0)
                return local_obs_t

        # Dict: 取第一个 value 当作本地观测
        if isinstance(obs, dict):
            for _, value in obs.items():
                if isinstance(value, (torch.Tensor, np.ndarray, list)):
                    local_obs = value
                    if isinstance(local_obs, torch.Tensor):
                        local_obs_t = local_obs.float()
                    else:
                        local_obs_t = torch.as_tensor(local_obs, dtype=torch.float32)
                    if local_obs_t.dim() == 1:
                        local_obs_t = local_obs_t.unsqueeze(0)
                    return local_obs_t

        # 默认情况
        local_obs_t = torch.as_tensor(obs, dtype=torch.float32)
        if local_obs_t.dim() == 1:
            local_obs_t = local_obs_t.unsqueeze(0)
        return local_obs_t

    def forward(self, input_dict, state, seq_lens):
        """
        IPPO 前向传播：
        - Actor：使用本地观测
        - Critic：同样使用本地观测（分散式）
        """
        obs = input_dict["obs"]

        # 提取本地观测
        local_obs = self._extract_local_obs(obs)
        if local_obs.dim() > 2:
            local_obs = local_obs.view(local_obs.size(0), -1)

        # 关键修正：如果实际本地观测维度与 self.local_obs_dim 不一致，进行扩展/截断
        in_dim = local_obs.size(-1)
        if in_dim != self.local_obs_dim:
            if self.local_obs_dim % in_dim == 0:
                repeat_times = self.local_obs_dim // in_dim
                local_obs = local_obs.repeat(1, repeat_times)
            elif in_dim > self.local_obs_dim:
                local_obs = local_obs[..., : self.local_obs_dim]
            else:
                pad_size = self.local_obs_dim - in_dim
                pad = torch.zeros(
                    local_obs.size(0), pad_size, device=local_obs.device, dtype=local_obs.dtype
                )
                local_obs = torch.cat([local_obs, pad], dim=-1)

        # Actor网络
        logits = self.actor_net(local_obs)  # [B, action_dim]

        # Critic网络（分散式，只用本地观测）
        value = self.critic_net(local_obs).squeeze(-1)  # [B]

        self._value_out = value
        self._local_obs = local_obs

        return logits, state

    def value_function(self):
        """RLlib 调用的 Critic 价值输出"""
        if self._value_out is None:
            return torch.zeros(1)
        return self._value_out

    def get_local_obs(self):
        """可选：外部获取当前批次的本地观测"""
        return self._local_obs


# 注册 IPPO 模型到 RLlib
ModelCatalog.register_custom_model("ippo_model", IndependentPPOModel)

# ==================== IPPO 模型定义结束 ====================


def train_balance_ippo_gpu():
    """使用与 mappo 相同超参数的 IPPO 风格训练 balance 场景 - GPU 版"""

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
            "kl_coeff": 0.5,
            "lambda": 0.95,
            "clip_param": 0.1,
            "vf_loss_coeff": 1.0,
            "entropy_coeff": 0.02,

            "train_batch_size": 4000,
            "rollout_fragment_length": 400,
            "sgd_minibatch_size": 64,
            "num_sgd_iter": 5,

            "num_gpus": 1,
            "num_workers": num_workers,
            "num_envs_per_worker": num_vectorized_envs,

            "lr": 2e-4,
            "gamma": 0.99,
            "use_gae": True,
            "batch_mode": "truncate_episodes",

            # === 使用自定义 IPPO 模型 ===
            "model": {
                "custom_model": "ippo_model",
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
                "device": vmas_device,
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
        train_balance_ippo_gpu()
    finally:
        if ray.is_initialized():
            ray.shutdown()