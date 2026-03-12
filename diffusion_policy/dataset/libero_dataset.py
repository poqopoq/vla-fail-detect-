import sys
sys.path.append("/home/zhiyuanjia/FAIL-Detect")

from diffusion_policy.dataset.robomimic_replay_image_dataset_for_libero import RobomimicReplayImageDataset

shape_meta = {
  'action': {
    'shape': [10]
  },
  'obs': {
    'agentview_rgb': {
      'shape': [3, 128, 128],
      'type': 'rgb'
    },
    'ee_pos': {
      'shape': [3]
    },
    'ee_ori': {
      'shape': [3]
    },
    'eye_in_hand_rgb': {
      'shape': [3, 128, 128],
      'type': 'rgb'
    },
    'gripper_states': {
      'shape': [2]
    }
  }
}

dataset_path = "/home/zhiyuanjia/LIBERO/datasets/libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_demo.hdf5"

dataset = RobomimicReplayImageDataset(
    shape_meta=shape_meta,
    dataset_path=dataset_path,
    horizon=16,
    abs_action=True,
    n_obs_steps=2,
    pad_after=7,
    pad_before=1,
    rotation_rep='rotation_6d',
    seed=42,
    use_cache=True,
    val_ratio=0.02
)

sample = dataset[0]

obs = sample["obs"]
action = sample["action"]

print("\n===== Observation keys =====")
for k in obs:
    print(k, obs[k].shape)

print("\n===== Action =====")
print(action.shape)

from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=4, shuffle=True)

batch = next(iter(loader))

print("\n===== Batch Shapes =====")

for k in batch["obs"]:
    print("obs", k, batch["obs"][k].shape)

print("action", batch["action"].shape)