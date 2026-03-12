# save_hdf5_to_pt_allinone.py
import pathlib, sys, os, h5py, torch
from diffusers.schedulers import DDPMScheduler
from diffusion_policy.policy.flow_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply

# FAIL-Detect 根目录
ROOT_DIR = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_DIR))


def load_hdf5(hdf5_path, obs_keys=None):
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


def generate_pt_allinone(hdf5_path, output_file, obs_keys=None, n_obs_steps=2, batch_size=16, device='cpu'):
    device = torch.device(device)

    # 1️⃣ 读取 HDF5
    obs_list, actions_list = load_hdf5(hdf5_path, obs_keys)
    print(f"Loaded {len(obs_list)} demos")

    # 2️⃣ 初始化 obs_encoder
    scheduler = DDPMScheduler(num_train_timesteps=100, beta_start=0.0001, beta_end=0.02, beta_schedule="squaredcos_cap_v2")
    shape_meta = {
        'action': {'shape':[actions_list[0].shape[1]]},
        'obs': {k:{'shape':[-1]} for k in (obs_keys or [])}  # 简化 shape
    }
    model = DiffusionUnetHybridImagePolicy(
        shape_meta=shape_meta,
        noise_scheduler=scheduler,
        horizon=16,
        n_action_steps=actions_list[0].shape[1],
        n_obs_steps=n_obs_steps
    ).to(device)
    model.eval()

    # 3️⃣ 遍历 demo -> batch -> obs_encoder -> 拼接
    all_x, all_y = [], []
    for demo_idx, (obs_demo, actions_demo) in enumerate(zip(obs_list, actions_list)):
        with torch.no_grad():
            for i in range(0, obs_demo.shape[0], batch_size):
                batch_obs = obs_demo[i:i+batch_size].to(device)
                batch_actions = actions_demo[i:i+batch_size]

                # normalize obs
                nobs = model.normalizer.normalize({'obs': batch_obs})
                this_nobs = dict_apply(nobs, lambda x: x[:,:n_obs_steps,...].reshape(-1,*x.shape[2:]))

                # obs_encoder
                features = model.obs_encoder(this_nobs)
                global_cond = features.reshape(batch_actions.shape[0], -1)

                all_x.append(global_cond.cpu())
                all_y.append(batch_actions)

        print(f"Processed demo {demo_idx+1}/{len(obs_list)}")

    # 拼接所有 demo
    X = torch.cat(all_x, dim=0)
    Y = torch.cat(all_y, dim=0)
    torch.save({'X': X, 'Y': Y}, output_file)
    print(f"Saved all demos into {output_file}, X {X.shape}, Y {Y.shape}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--n_obs_steps", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--obs_keys", nargs='+', default=['agentview_rgb','ee_pos','ee_ori','eye_in_hand_rgb','gripper_states'])
    args = parser.parse_args()

    generate_pt_allinone(args.hdf5, args.output_file, args.obs_keys, args.n_obs_steps, args.batch_size, args.device)