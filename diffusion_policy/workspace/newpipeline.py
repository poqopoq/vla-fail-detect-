# newpipeline_split_optimized.py
import pathlib
import sys
import h5py
import torch
import argparse
import os
from diffusion_policy.policy.flow_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
from diffusers.schedulers import DDPMScheduler

# 将 FAIL-Detect 根目录加入 Python 路径
ROOT_DIR = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_DIR))

def adjust_xshape(x, in_dim):
    """调整 X 维度，使其总长度能被 in_dim 整除并且 reshape 后长度为 4 的倍数"""
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

def infer_obs_keys(hdf5_path):
    """自动获取 HDF5 内 demo 第一个 obs group 的 key"""
    with h5py.File(hdf5_path, 'r') as f:
        first_demo = list(f['data'].keys())[0]
        obs_keys = list(f['data'][first_demo]['obs'].keys())
    return obs_keys

def main(hdf5_path, output_dir, n_obs_steps=2, batch_size=64, device_str=None, max_steps=None):
    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(output_dir, exist_ok=True)

    # 自动获取 obs_keys
    obs_keys = infer_obs_keys(hdf5_path)
    print(f"Using obs keys: {obs_keys}")

    # 读取 HDF5 分 demo 处理
    with h5py.File(hdf5_path, 'r') as f:
        data_group = f['data']
        for demo_idx, demo_key in enumerate(data_group.keys()):
            demo = data_group[demo_key]
            actions_demo = torch.from_numpy(demo['actions'][:]).float()
            obs_group = demo['obs']

            # 限制 demo 最大步长
            if max_steps is not None and actions_demo.shape[0] > max_steps:
                actions_demo = actions_demo[:max_steps]
            
            obs_tensors = []
            for k in obs_keys:
                v = obs_group[k][:]
                if max_steps is not None and v.shape[0] > max_steps:
                    v = v[:max_steps]
                v = torch.from_numpy(v).float()
                if len(v.shape) > 2:  # 展平图像
                    v = v.reshape(v.shape[0], -1)
                obs_tensors.append(v)
            obs_demo = torch.cat(obs_tensors, dim=1)

            # 初始化 Flow UNet (只用于 obs_encoder)
            scheduler = DDPMScheduler(
                num_train_timesteps=100,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule="squaredcos_cap_v2"
            )
            shape_meta = {
                'action': {'shape':[actions_demo.shape[1]]},
                'obs': {k: {'shape': obs_demo.shape[1:]} for k in obs_keys}
            }
            model = DiffusionUnetHybridImagePolicy(
                shape_meta=shape_meta,
                noise_scheduler=scheduler,
                horizon=16,
                n_action_steps=actions_demo.shape[1],
                n_obs_steps=n_obs_steps,
                obs_as_global_cond=True,
                crop_shape=(76,76),
                diffusion_step_embed_dim=128,
                down_dims=(512,1024,2048),
                kernel_size=5,
                n_groups=8,
                cond_predict_scale=True,
                obs_encoder_group_norm=True,
                eval_fixed_crop=True
            ).to(device)

            # 分 batch 编码并保存
            X_demo_list, Y_demo_list = [], []
            with torch.no_grad():
                for i in range(0, obs_demo.shape[0], batch_size):
                    batch = obs_demo[i:i+batch_size].to(device)
                    X_batch = model.obs_encoder(batch).cpu()
                    Y_batch = actions_demo[i:i+batch_size]
                    X_demo_list.append(X_batch)
                    Y_demo_list.append(Y_batch)

            X_demo = torch.cat(X_demo_list, dim=0)
            X_demo = adjust_xshape(X_demo, in_dim=actions_demo.shape[1])
            Y_demo = torch.cat(Y_demo_list, dim=0)
            demo_path = os.path.join(output_dir, f'demo{demo_idx}.pt')
            torch.save({'X': X_demo, 'Y': Y_demo}, demo_path)
            print(f"Saved demo {demo_idx} -> {demo_path}, X: {X_demo.shape}, Y: {Y_demo.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_obs_steps", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=None, help="最大 demo 步长，防止内存爆炸")
    args = parser.parse_args()
    main(args.hdf5, args.output_dir, args.n_obs_steps, args.batch_size, args.device, args.max_steps)