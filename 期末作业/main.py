import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def plot_training_visualization(csv_file_path):
    """基于训练数据创建可视化图表"""
    df = pd.read_csv(csv_file_path)

    # 创建综合训练可视化
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Balance 任务训练过程可视化', fontsize=16)

    # 1. 奖励变化
    axes[0, 0].plot(df['training_iteration'], df['episode_reward_mean'],
                    'b-', linewidth=2, label='平均奖励')
    axes[0, 0].fill_between(df['training_iteration'],
                            df['episode_reward_min'],
                            df['episode_reward_max'],
                            alpha=0.2, color='blue', label='奖励范围')
    axes[0, 0].set_xlabel('训练迭代')
    axes[0, 0].set_ylabel('奖励值')
    axes[0, 0].set_title('奖励值变化曲线')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # 2. 移动平均奖励
    window_size = max(1, len(df) // 20)  # 动态窗口大小
    df['reward_ma'] = df['episode_reward_mean'].rolling(window=window_size, center=True).mean()
    axes[0, 1].plot(df['training_iteration'], df['episode_reward_mean'],
                    alpha=0.5, label='原始奖励', color='lightblue')
    axes[0, 1].plot(df['training_iteration'], df['reward_ma'],
                    label=f'{window_size}轮移动平均', linewidth=2, color='red')
    axes[0, 1].set_xlabel('训练迭代')
    axes[0, 1].set_ylabel('奖励值')
    axes[0, 1].set_title('奖励值及移动平均')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # 3. 回合长度
    axes[0, 2].plot(df['training_iteration'], df['episode_len_mean'],
                    'green', linewidth=2)
    axes[0, 2].set_xlabel('训练迭代')
    axes[0, 2].set_ylabel('回合长度')
    axes[0, 2].set_title('平均回合长度变化')
    axes[0, 2].grid(True, alpha=0.3)

    # 4. 性能热力图
    performance_data = df[['episode_reward_mean', 'episode_len_mean']].values.T
    im = axes[1, 0].imshow(performance_data, aspect='auto', cmap='viridis', origin='lower')
    axes[1, 0].set_xlabel('训练迭代')
    axes[1, 0].set_ylabel('指标')
    axes[1, 0].set_yticks([0, 1])
    axes[1, 0].set_yticklabels(['平均奖励', '回合长度'])
    axes[1, 0].set_title('性能热力图')
    plt.colorbar(im, ax=axes[1, 0])

    # 5. 训练效率
    axes[1, 1].scatter(df['training_iteration'], df['time_this_iter_s'],
                       alpha=0.6, s=20)
    axes[1, 1].set_xlabel('训练迭代')
    axes[1, 1].set_ylabel('每轮耗时 (秒)')
    axes[1, 1].set_title('训练效率')
    axes[1, 1].grid(True, alpha=0.3)

    # 6. 奖励分布
    axes[1, 2].hist(df['episode_reward_mean'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 2].axvline(df['episode_reward_mean'].mean(), color='red', linestyle='--',
                       label=f'平均值: {df["episode_reward_mean"].mean():.2f}')
    axes[1, 2].set_xlabel('奖励值')
    axes[1, 2].set_ylabel('频率')
    axes[1, 2].set_title('奖励值分布')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 打印训练总结
    print("\n=== 训练总结 ===")
    print(f"训练迭代次数: {len(df)}")
    print(f"初始平均奖励: {df['episode_reward_mean'].iloc[0]:.3f}")
    print(f"最终平均奖励: {df['episode_reward_mean'].iloc[-1]:.3f}")
    print(f"奖励改善幅度: {df['episode_reward_mean'].iloc[-1] - df['episode_reward_mean'].iloc[0]:.3f}")
    print(f"最高平均奖励: {df['episode_reward_mean'].max():.3f}")
    print(f"奖励标准差: {df['episode_reward_mean'].std():.3f}")
    print(
        f"训练稳定性: {'优秀' if df['episode_reward_mean'].std() < 3 else '良好' if df['episode_reward_mean'].std() < 5 else '一般'}")
    # 重点查看 reward 断崖处附近，比如 90~120 轮

    sub = df[(df['training_iteration'] >= 90) & (df['training_iteration'] <= 120)]
    cols = [
        'training_iteration',
        'episode_reward_mean',
        'episode_reward_max',
        'episode_reward_min',
        'episode_len_mean',
        'info/learner/default_policy/learner_stats/kl',
        'info/learner/default_policy/learner_stats/total_loss',
        'info/learner/default_policy/learner_stats/vf_loss',
        'info/learner/default_policy/learner_stats/grad_gnorm',
        'info/learner/default_policy/learner_stats/entropy',
    ]
    print(sub[cols])

# 使用 CSV 数据进行可视化
#csv_path = "C:/Users/Lenovo/ray_results/PPO_2026-01-13_22-57-01/PPO_balance_1dfd6_00000_0_2026-01-13_22-57-01/progress.csv"
#csv_path = "C:/Users/Lenovo/ray_results/PPO_2026-01-14_09-20-31/PPO_balance_3819a_00000_0_2026-01-14_09-20-31/progress.csv"
#csv_path ="C:/Users/Lenovo/ray_results/PPO_2026-01-14_17-15-14/PPO_balance_88ebb_00000_0_2026-01-14_17-15-14/progress.csv"
# csv_path ="C:/Users/Lenovo/ray_results/PPO_2026-01-15_00-34-25/PPO_balance_e353b_00000_0_2026-01-15_00-34-25/progress.csv"
#csv_path ="C:/Users/Lenovo/ray_results/PPO_2026-01-15_11-49-01/PPO_balance_2167d_00000_0_2026-01-15_11-49-02/progress.csv"
#csv_path = "PPO01.csv"
#csv_path = "PPO02.csv"
#csv_path = "CPPO.csv"
#csv_path = "MAPPO.csv"
#csv_path = "IPPO.csv"
plot_training_visualization(csv_path)