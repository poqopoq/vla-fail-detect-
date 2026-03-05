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
