import h5py
import torch
import numpy as np
from tqdm import tqdm

file = "dataset.hdf5"

f = h5py.File(file,'r')

X_list = []
Y_list = []

demos = list(f["data"].keys())

for demo in tqdm(demos):

    data = f["data"][demo]

    actions = data["actions"][:]        # (T,7)

    obs = data["obs"]

    ee_pos = obs["ee_pos"][:]           # (T,3)
    ee_ori = obs["ee_ori"][:]           # (T,4)
    gripper = obs["gripper_states"][:]  # (T,2)
    joint = obs["joint_states"][:]      # (T,?)

    # 拼 observation
    state = np.concatenate([
        ee_pos,
        ee_ori,
        gripper,
        joint
    ],axis=1)

    X_list.append(state)
    Y_list.append(actions)

X = np.concatenate(X_list,axis=0)
Y = np.concatenate(Y_list,axis=0)

X = torch.tensor(X).float()
Y = torch.tensor(Y).float()

print("X shape:",X.shape)
print("Y shape:",Y.shape)

torch.save({
    "X":X,
    "Y":Y
},"square_data_flow.pt")