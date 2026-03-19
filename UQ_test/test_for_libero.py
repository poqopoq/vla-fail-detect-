import torch
import numpy as np
import matplotlib.pyplot as plt
import eval_load_baseline as elb  

# ---------------------------
# 配置
# ---------------------------
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
task_name = 'square'          
policy_type = 'diffusion'          
data_path = '/home/zhiyuanjia/FAIL-Detect/outputs/2026-03-19/04-07-51/demo0_fail.pt'     
max_trajectories = 5          # 只画前几条轨迹避免太多重叠

# ---------------------------
# 1️⃣ 读取数据
# ---------------------------
data = torch.load(data_path)
obs_tensor = data['X'].to(device)  # [N, flat_dim]
action_tensor = data['Y'].to(device)  # [N, flat_dim]

N = obs_tensor.shape[0]          # 轨迹数
in_dim_dict = {'square': 10, 'transport': 20, 'tool_hang': 20, 'can': 10}
in_dim = in_dim_dict[task_name]  # 每个时间步维度

# ---------------------------
# 2️⃣ 加载 logpZO baseline 模型
# ---------------------------
baseline_model = elb.get_baseline_model('logpZO', task_name, policy_type).to(device)
baseline_model.eval()
baseline_model.global_eps = None
print("Loaded logpZO baseline model")

# ---------------------------
# 3️⃣ 计算 logpZO 不确定性
# ---------------------------
uq_values = elb.logpZO_UQ(baseline_model, obs_tensor, action_pred=None, task_name=task_name)
uq_values = uq_values.cpu().numpy()  # [N, flat_dim_after_adjust]

# 自动计算每条轨迹的时间步长度 T
flat_dim = uq_values.shape[1]
T = flat_dim // in_dim  # 时间步数
print(f"Number of trajectories: {N}, Time steps per trajectory: {T}, dim per step: {in_dim}")

# ---------------------------
# 4️⃣ 绘制折线图
# ---------------------------
plt.figure(figsize=(12,6))
for i in range(min(N, max_trajectories)):
    # reshape 回 (T, in_dim)，这里用每时间步的平均值表示
    traj = uq_values[i].reshape(T, in_dim).mean(axis=1)
    plt.plot(range(T), traj, label=f'Trajectory {i+1}')

plt.xlabel('Time Step')
plt.ylabel('logpZO Uncertainty')
plt.title(f'logpZO Uncertainty per Time Step ({task_name})')
plt.grid(True)
plt.legend()
plt.show()