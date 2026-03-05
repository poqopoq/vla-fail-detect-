
import h5py
import torch
from torch.utils.data import DataLoader, TensorDataset
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# -------------------------
# 1. 配置参数
# -------------------------
HDF5_FILE = '/home/zhiyuanjia/LIBERO/datasets/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5'  # 改成你的文件
N_OBS_STEPS = 2
N_ACTION_STEPS = 8
HORIZON = 16
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 用于 DiffusionUnetHybridImagePolicy 初始化的 shape_meta 和 noise_scheduler
shape_meta = {
    "action": {"shape": [7]},
    "obs": {
        "agentview_rgb": {"shape": [3, 84, 84], "type": "rgb"},
        "ee_pos": {"shape": [3]},
        "ee_ori": {"shape": [4]},
        "eye_in_hand_rgb": {"shape": [3, 84, 84], "type": "rgb"},
        "gripper_states": {"shape": [2]},
        "ee_states": {"shape": [3]},
        "joint_states": {"shape": [7]}
    }
}

noise_scheduler = DDPMScheduler(
    beta_start=0.0001,
    beta_end=0.02,
    beta_schedule="squaredcos_cap_v2",
    clip_sample=True,
    num_train_timesteps=100,
    prediction_type="epsilon",
    variance_type="fixed_small"
)

# -------------------------
# 2. 初始化 obs_encoder
# -------------------------
tasks = {10: 'square', 20: 'transport', 10: 'tool_hang', 10: 'lift', 10: 'can', 7: 'libero'}

model = DiffusionUnetHybridImagePolicy(
    shape_meta=shape_meta,
    noise_scheduler=noise_scheduler,
    horizon=HORIZON,
    n_action_steps=N_ACTION_STEPS,
    n_obs_steps=N_OBS_STEPS
)
model.obs_encoder.to(DEVICE)
model.obs_encoder.eval()  # 不训练

# -------------------------
# 3. 读取 HDF5
# -------------------------
all_obs = []
all_actions = []

with h5py.File(HDF5_FILE, "r") as f:
    for demo_key in f["data"]:
        demo = f["data"][demo_key]
        obs_group = demo["obs"]
        # 这里把需要的 obs 都按顺序拼接成字典
        obs_dict = {
            "agentview_rgb": torch.tensor(obs_group["agentview_rgb"][()]).float(),
            "ee_pos": torch.tensor(obs_group["ee_pos"][()]).float(),
            "ee_ori": torch.tensor(obs_group["ee_ori"][()]).float(),
            "eye_in_hand_rgb": torch.tensor(obs_group["eye_in_hand_rgb"][()]).float(),
            "gripper_states": torch.tensor(obs_group["gripper_states"][()]).float(),
            "ee_states": torch.tensor(obs_group["ee_states"][()]).float(),
            "joint_states": torch.tensor(obs_group["joint_states"][()]).float()
        }
        actions = torch.tensor(demo["actions"][()]).float()

        # 如果 obs 维度有 batch 维
        for k in obs_dict:
            obs_dict[k] = obs_dict[k].unsqueeze(0) if len(obs_dict[k].shape) == 1 else obs_dict[k]

        all_obs.append(obs_dict)
        all_actions.append(actions)

# -------------------------
# 4. DataLoader
# -------------------------
# 这里简单把每个 demo 拼接起来
obs_batch_list = {k: torch.cat([d[k] for d in all_obs], dim=0) for k in all_obs[0]}
actions_tensor = torch.cat(all_actions, dim=0)

dataset = TensorDataset(torch.arange(len(actions_tensor)))  # dummy dataset for batching
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# -------------------------
# 5. 遍历生成 X/Y
# -------------------------
full_X = []
full_Y = []

with torch.no_grad():
    for idxs in dataloader:
        batch_idx = idxs[0]
        batch_obs = {k: v[batch_idx].to(DEVICE) for k, v in obs_batch_list.items()}
        # 通过 obs_encoder 得到特征
        B = batch_obs["ee_pos"].shape[0]
        obs_list = [v.unsqueeze(1) if len(v.shape) == 3 else v for v in batch_obs.values()]
        # 调用 obs_encoder
        obs_tensor = batch_obs["agentview_rgb"].to(DEVICE)  # 默认第一个 RGB 输入，其他低维单独处理
        # 可以根据原作者 obs_encoder 逻辑改
        features = model.obs_encoder(obs_tensor)
        global_cond = features.reshape(B, -1)

        full_X.append(global_cond.cpu())
        full_Y.append(actions_tensor[batch_idx].cpu())

full_X = torch.cat(full_X, dim=0)
full_Y = torch.cat(full_Y, dim=0)

# -------------------------
# 6. 保存 .pt
# -------------------------
torch.save({"X": full_X, "Y": full_Y}, "libero_data.pt")
print("Saved X/Y to libero_data.pt")
print("X shape:", full_X.shape, "Y shape:", full_Y.shape)