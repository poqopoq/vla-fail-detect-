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
data_path = '../../data/outputs/square_data_diffusion.pt'     
max_trajectories = 5          # 只画前几条轨迹避免太多重叠

# adjust_xshape used in logpZO baseline
def adjust_xshape(x, in_dim):
    total_dim = x.shape[1]

    remain_dim = total_dim % in_dim
    if remain_dim > 0:
        pad = in_dim - remain_dim
        x = torch.cat([x, torch.zeros(x.shape[0], pad, device=x.device)], dim=1)
        total_dim += pad

    reshaped_dim = total_dim // in_dim
    if reshaped_dim % 4 != 0:
        extra_pad = (4 - (reshaped_dim % 4)) * in_dim
        x = torch.cat([x, torch.zeros(x.shape[0], extra_pad, device=x.device)], dim=1)

    return x.reshape(x.shape[0], -1, in_dim)

# logpZO baseline 

def logpZO_UQ(baseline_model, observation, action_pred=None, task_name='square'):
    in_dim = 10
    if task_name == 'transport':
        in_dim = 20

    observation = adjust_xshape(observation, in_dim)

    if action_pred is not None:
        observation = torch.cat([observation, action_pred], dim=1)

    with torch.no_grad():
        timesteps = torch.zeros(observation.shape[0], device=observation.device)
        pred_v = baseline_model(observation, timesteps)
        observation = observation + pred_v
        logpZO = observation.reshape(len(observation), -1).pow(2).sum(dim=-1)

    return logpZO

# ---------------------------
# load test data
# ---------------------------
data = torch.load(data_path)
obs_tensor = data['X'].to(device)  # [N, T, obs_dim]
action_tensor = data['Y'].to(device)  # [N, T, action_dim]

# ---------------------------
# load logpZO baseline model
# ---------------------------
baseline_model = elb.get_baseline_model('logpZO', task_name, policy_type).to(device)
baseline_model.eval()
baseline_model.global_eps = None
print("Loaded logpZO baseline model")

# ---------------------------
# 3️⃣ 计算 logpZO 不确定性
# ---------------------------
uq_values = elb.logpZO_UQ(baseline_model, obs_tensor, action_pred=None, task_name=task_name)
uq_values = uq_values.reshape(N, T).cpu().numpy()  # reshape 回轨迹形状

# ---------------------------
# 4️⃣ 绘制折线图
# ---------------------------
plt.figure(figsize=(12,6))
for i in range(min(N, max_trajectories)):
    plt.plot(range(T), uq_values[i], label=f'Trajectory {i+1}')

plt.xlabel('Time Step')
plt.ylabel('logpZO Uncertainty')
plt.title(f'logpZO Uncertainty per Time Step ({task_name})')
plt.grid(True)
plt.legend()
plt.show()