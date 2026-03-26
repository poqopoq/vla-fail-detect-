import torch
import numpy as np
import matplotlib.pyplot as plt
import eval_load_baseline as elb  


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
task_name = 'square'          
policy_type = 'diffusion'          
data_path = '/home/zhiyuanjia/FAIL-Detect/outputs/2026-03-19/04-07-51/demo0_fail.pt'     
batch_size = 16


data = torch.load(data_path)
obs_tensor = data['X'].to(device)  # shape [num_samples, obs_feature_dim_flat]
action_tensor = data['Y'].to(device)

num_samples = obs_tensor.shape[0]


baseline_model = elb.get_baseline_model('logpZO', task_name, policy_type).to(device)
baseline_model.eval()
baseline_model.global_eps = None
print("Loaded logpZO baseline model")


uq_values_list = []

for start in range(0, num_samples, batch_size):
    end = min(start + batch_size, num_samples)
    batch = obs_tensor[start:end]  # shape [B, obs_feature_dim_flat]

    
    uq_batch = elb.logpZO_UQ(baseline_model, batch, action_pred=None, task_name=task_name)
    
    uq_values_list.append(uq_batch.cpu().numpy())


uq_values = np.concatenate(uq_values_list, axis=0)  # shape [num_samples, ]



if uq_values.ndim > 1:
    uq_ordered = uq_values.mean(axis=1)  # shape [num_samples,]
else:
    uq_ordered = uq_values   

plt.figure(figsize=(12,6))
plt.plot(range(len(uq_ordered)), uq_ordered, label='logpZO Uncertainty')
plt.xlabel('Time Step / Sample Index')
plt.ylabel('logpZO Uncertainty')
plt.title(f'logpZO Uncertainty over Time ({task_name})')
plt.grid(True)
plt.legend()
plt.show()


prediction_trajectory = regress(training_data, self.regression_type)