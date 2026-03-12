/home/zhiyuanjia/FAIL-Detect/diffusion_policy/workspace/train_diffusion_unet_hybrid_workspace_get_data.py这里面包含了原始hdf5转可训练用.pt文件方法
转liberohdf5文件到。pt

x和y是从哪里来的
Y ： abs_action
task:
  abs_action: true
  dataset:
    dataset_path: data/robomimic/datasets/square/ph/image_abs.hdf5
    n_obs_steps: 2
    shape_meta:
      action:
        shape:
        - 10


X = obs_encoder(obs[:, :n_obs_steps, ...])

policy:
  obs_as_global_cond: true
  n_obs_steps: 2
  obs_encoder_group_norm: true

obser: 
shape_meta:
  obs:
    agentview_image:
      shape: [3, 84, 84]
    robot0_eef_pos:
      shape: [3]
    robot0_eef_quat:
      shape: [4]
    robot0_eye_in_hand_image:
      shape: [3, 84, 84]
    robot0_gripper_qpos:
      shape: [2]


python benchmark_scripts/render_single_task.py --benchmark_name libero_10 --task_id 0 libero 渲染运行结果


/home/zhiyuanjia/FAIL-Detect/diffusion_policy/workspace/train_diffusion_unet_hybrid_workspace_get_data.py描述了怎么hdf5处理为.pt

/home/zhiyuanjia/FAIL-Detect/diffusion_policy/policy/diffusion_transformer_hybrid_image_policy.py obs_encoder所在处


训练policy 时也训练了obs_encoder 但是没有只能随机


obs 拆分送入encoder


/home/zhiyuanjia/FAIL-Detect/diffusion_policy/policy/diffusion_transformer_hybrid_image_policy.py obs_encoder


| 你的 key            | shape              | dtype   | 原作者对应 key                  | 原作者期望 shape/dtype            |
| ----------------- | ------------------ | ------- | -------------------------- | ---------------------------- |
| `agentview_rgb`   | (272, 128, 128, 3) | uint8   | `agentview_image`          | (T, 3, 84, 84), float32, 0-1 |
| `eye_in_hand_rgb` | (272, 128, 128, 3) | uint8   | `robot0_eye_in_hand_image` | (T, 3, 84, 84), float32, 0-1 |
| `ee_pos`          | (272, 3)           | float64 | `robot0_eef_pos`           | (T, 3), float32              |
| `ee_ori`          | (272, 3)           | float64 | `robot0_eef_quat`          | (T, 4), float32 (原作者 4D)     |  需要RotationTransformer  rotation_rep='rotation_6d'
| `gripper_states`  | (272, 2)           | float64 | `robot0_gripper_qpos`      | (T, 2), float32              |
| `ee_states`       | (272, 6)           | float64 | —                          | 不用（可忽略）                      |
| `joint_states`    | (272, 7)           | float64 | —                          | 不用（可忽略）                      |
| `actions`         | (272, 7)           | float64 | `action`                   | (T, 10)                      |



dataset
   ↓
DataLoader    buffer_replay_image + 指定key obs_shape_meta = shape_meta['obs']
                                            for key, attr in obs_shape_meta.items():
                                                type = attr.get('type','low_dim')
                                                if type == 'rgb':
                                                    obs_config['rgb'].append(key)
                                                elif type == 'low_dim':
                                                    obs_config['low_dim'].append(key)
   ↓
batch = {obs, action}
   ↓
normalizer.normalize
   ↓
obs_encoder   obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
   ↓
feature X
   ↓
reshape action
   ↓
Y
   ↓
torch.save(X,Y)


Keys in demo_0: ['actions', 'dones', 'next_obs', 'obs', 'rewards', 'states']
actions: (127, 7), dtype=float64    action dimension 7 
dones: (127,), dtype=int64
next_obs: Group with keys ['object', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_eef_quat_site', 'robot0_gripper_qpos', 'robot0_gripper_qvel', 'robot0_joint_pos', 'robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel']
obs: Group with keys ['object', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_eef_quat_site', 'robot0_gripper_qpos', 'robot0_gripper_qvel', 'robot0_joint_pos', 'robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel']
  obs/object: (127, 14), dtype=float64
  obs/robot0_eef_pos: (127, 3), dtype=float64
  obs/robot0_eef_quat: (127, 4), dtype=float64
  obs/robot0_eef_quat_site: (127, 4), dtype=float32
  obs/robot0_gripper_qpos: (127, 2), dtype=float64
  obs/robot0_gripper_qvel: (127, 2), dtype=float64
  obs/robot0_joint_pos: (127, 7), dtype=float64
  obs/robot0_joint_pos_cos: (127, 7), dtype=float64
  obs/robot0_joint_pos_sin: (127, 7), dtype=float64
  obs/robot0_joint_vel: (127, 7), dtype=float64
rewards: (127,), dtype=float64
states: (127, 45), dtype=float64



python workspace/train_diffusion_unet_hybrid_workspace_get_data_libero.py   --config-path /home/zhiyuanjia/FAIL-Detect/diffusion_policy/configs_robomimic   --config-name image_square_ph_visual_diffusion_policy_cnn_for_libero