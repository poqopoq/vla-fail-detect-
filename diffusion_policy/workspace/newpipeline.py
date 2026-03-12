# newpipeline_full.py
import pathlib, sys, os, argparse, h5py, torch
from diffusers.schedulers import DDPMScheduler
from diffusion_policy.policy.flow_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy

# FAIL-Detect 根目录
ROOT_DIR = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_DIR))

def adjust_xshape(x, in_dim):
    """保证 reshape 后的维度可被4整除"""
    total_dim = x.shape[1]
    remain_dim = total_dim % in_dim
    if remain_dim > 0:
        pad = in_dim - remain_dim
        x = torch.cat([x, torch.zeros(x.shape[0], pad, device=x.device)], dim=1)
    reshaped_dim = x.shape[1] // in_dim
    if reshaped_dim % 4 != 0:
        extra_pad = (4 - (reshaped_dim % 4)) * in_dim
        x = torch.cat([x, torch.zeros(x.shape[0], extra_pad, device=x.device)], dim=1)
    return x.reshape(x.shape[0], -1, in_dim)

def load_hdf5(hdf5_path, obs_keys=None):
    """返回 list of obs_tensors, list of actions_tensors"""
    if obs_keys is None:
        obs_keys = ['agentview_rgb','ee_pos','ee_ori','eye_in_hand_rgb','gripper_states']
    obs_list, actions_list = [], []
    with h5py.File(hdf5_path,'r') as f:
        for demo_key in f['data'].keys():
            demo = f['data'][demo_key]
            actions = torch.from_numpy(demo['actions'][:]).float()
            obs_group = demo['obs']
            obs_tensors = []
            for k in obs_keys:
                v = torch.from_numpy(obs_group[k][:]).float()
                if len(v.shape) > 2:  # 展平图像
                    v = v.reshape(v.shape[0], -1)
                obs_tensors.append(v)
            obs_tensor = torch.cat(obs_tensors, dim=1)
            obs_list.append(obs_tensor)
            actions_list.append(actions)
    return obs_list, actions_list

def main(hdf5_path, output_dir, n_obs_steps=2, batch_size=64, device_str=None):
    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))

    # 读取 HDF5
    obs_list, actions_list = load_hdf5(hdf5_path)
    print(f"Loaded {len(obs_list)} demos")

    # 初始化 obs_encoder
    scheduler = DDPMScheduler(num_train_timesteps=100, beta_start=0.0001, beta_end=0.02, beta_schedule="squaredcos_cap_v2")
    shape_meta = {
        'action': {'shape':[actions_list[0].shape[1]]},
        'obs': {'agentview_image': {'shape':[3,84,84]},
                'robot0_eye_in_hand_image': {'shape':[3,84,84]},
                'robot0_eef_pos': {'shape':[3]},
                'robot0_eef_quat': {'shape':[4]},
                'robot0_gripper_qpos': {'shape':[2]}}
    }

    model = DiffusionUnetHybridImagePolicy(
        shape_meta=shape_meta,
        noise_scheduler=scheduler,
        horizon=16,
        n_action_steps=actions_list[0].shape[1],
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

    os.makedirs(output_dir, exist_ok=True)

    # 分 demo + batch 编码
    for demo_idx, obs_demo in enumerate(obs_list):
        actions_demo = actions_list[demo_idx]
        X_list, Y_list = [], []
        with torch.no_grad():
            for i in range(0, obs_demo.shape[0], batch_size):
                batch = obs_demo[i:i+batch_size].to(device)
                X_batch = model.obs_encoder(batch).cpu()
                Y_batch = actions_demo[i:i+batch_size]
                X_list.append(X_batch)
                Y_list.append(Y_batch)
        X_demo = torch.cat(X_list, dim=0)
        X_demo = adjust_xshape(X_demo, in_dim=actions_demo.shape[1])
        Y_demo = torch.cat(Y_list, dim=0)
        torch.save({'X': X_demo, 'Y': Y_demo}, os.path.join(output_dir,f'demo{demo_idx}.pt'))
        print(f"Saved demo {demo_idx}: X {X_demo.shape}, Y {Y_demo.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_obs_steps", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    main(args.hdf5, args.output_dir, args.n_obs_steps, args.batch_size, args.device)