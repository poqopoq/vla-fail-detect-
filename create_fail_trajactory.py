import h5py
import numpy as np
import os


def create_demo0_failure_state_obs(
    success_hdf5_path,
    failure_hdf5_path,
    state_noise_std=0.05
):
    """
    生成 demo_0 的失败轨迹：
    - 前半段 = 原轨迹
    - 后半段 actions = 完全随机（基于原 action 范围）
    - 后半段 states 加高斯噪声
    - 后半段 obs 按原分布随机生成
    """

    with h5py.File(success_hdf5_path, 'r') as f_in, \
         h5py.File(failure_hdf5_path, 'w') as f_out:

        data_out = f_out.create_group('data')
        demo_name = 'demo_0'

        grp_in = f_in['data'][demo_name]
        grp_out = data_out.create_group(demo_name)

        # ===== 读取原数据 =====
        actions = np.array(grp_in['actions'])
        states = np.array(grp_in['states'])
        rewards = np.array(grp_in['rewards'])
        robot_states = np.array(grp_in['robot_states'])
        dones = np.array(grp_in['dones'])
        obs_in = grp_in['obs']

        total_steps = len(actions)
        pre_steps = 0
        post_steps = total_steps - pre_steps

        # ===== 前半段 =====
        actions_pre = actions[:pre_steps]
        states_pre = states[:pre_steps]
        rewards_pre = rewards[:pre_steps]
        robot_states_pre = robot_states[:pre_steps]
        dones_pre = dones[:pre_steps]

        # ===== 后半段 =====

        # --- actions: 完全随机（基于原范围）---
        action_min = actions.min(axis=0)
        action_max = actions.max(axis=0)

        actions_post = np.random.uniform(
            low=action_min,
            high=action_max,
            size=(post_steps,) + actions.shape[1:]
        ).astype(actions.dtype)

        # --- states: 加噪声（更稳 shape 写法）---
        states_post = states[pre_steps:] + np.random.normal(
            0,
            state_noise_std,
            size=states[pre_steps:].shape
        )

        # --- 其他保持 ---
        rewards_post = rewards[pre_steps:]
        robot_states_post = robot_states[pre_steps:]
        dones_post = dones[pre_steps:]

        # ===== 拼接 =====
        actions_fail = np.concatenate([actions_pre, actions_post], axis=0)
        states_fail = np.concatenate([states_pre, states_post], axis=0)
        rewards_fail = np.concatenate([rewards_pre, rewards_post], axis=0)
        robot_states_fail = np.concatenate([robot_states_pre, robot_states_post], axis=0)

        # dones：保证最后结束
        dones_fail = np.concatenate([dones_pre, dones_post], axis=0)
        dones_fail[:] = False
        dones_fail[-1] = True

        # ===== 保存 =====
        grp_out.create_dataset('actions', data=actions_fail)
        grp_out.create_dataset('states', data=states_fail)
        grp_out.create_dataset('rewards', data=rewards_fail)
        grp_out.create_dataset('robot_states', data=robot_states_fail)
        grp_out.create_dataset('dones', data=dones_fail)

        # ===== obs =====
        obs_out = grp_out.create_group('obs')

        for key in obs_in.keys():
            data = np.array(obs_in[key])

            data_pre = data[:pre_steps]
            shape_post = (post_steps,) + data.shape[1:]

            if np.issubdtype(data.dtype, np.integer):
                # 图像类
                data_post = np.random.randint(
                    0, 256,
                    size=shape_post,
                    dtype=data.dtype
                )
            else:
                # 用原分布生成（更合理）
                mean = data.mean()
                std = data.std() + 1e-6
                data_post = np.random.normal(
                    mean,
                    std,
                    size=shape_post
                ).astype(data.dtype)

            data_fail = np.concatenate([data_pre, data_post], axis=0)
            obs_out.create_dataset(key, data=data_fail)

        print(f"{demo_name} converted to failure trajectory (random actions in second half).")

    print(f"Failure demo saved to {failure_hdf5_path}")


# ===== 运行入口 =====
if __name__ == "__main__":
    success_hdf5_path = "/home/zhiyuanjia/LIBERO/datasets/libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_demo.hdf5"

    failure_hdf5_path = "/home/zhiyuanjia/LIBERO/datasets/libero_10/SCENE1_failure_demo0_action.hdf5"

    create_demo0_failure_state_obs(
        success_hdf5_path,
        failure_hdf5_path,
        state_noise_std=0.5
    )