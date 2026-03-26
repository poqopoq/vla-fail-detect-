if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import numpy as np
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.pytorch_util import dict_apply

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDiffusionUnetHybridWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetHybridImagePolicy = hydra.utils.instantiate(cfg.policy)
        self.ema_model: DiffusionUnetHybridImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

def run(self):
    cfg = copy.deepcopy(self.cfg)

    # -----------------------------
    # 1. 数据集
    # -----------------------------
    dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
    assert isinstance(dataset, BaseImageDataset)
    train_dataloader = DataLoader(dataset, **cfg.dataloader)
    normalizer = dataset.get_normalizer()

    device = torch.device(cfg.training.device)
    self.ema_model.set_normalizer(normalizer)
    self.ema_model.to(device)

    # -----------------------------
    # 2. 滑动窗口生成 global_cond_seq 的函数
    # -----------------------------
    def build_sliding_window_cond(nobs: dict, model, mode="flatten"):
        B = list(nobs.values())[0].shape[0]
        T = list(nobs.values())[0].shape[1]
        K = model.n_obs_steps
        cond_list = []

        for t in range(T - K + 1):
            window = dict_apply(
                nobs,
                lambda x: x[:, t:t+K, ...]
            )
            window = dict_apply(
                window,
                lambda x: x.reshape(-1, *x.shape[2:])
            )
            feat = model.obs_encoder(window)
            feat = feat.reshape(B, K, -1)
            if mode == "flatten":
                cond = feat.reshape(B, -1)
            elif mode == "last":
                cond = feat[:, -1, :]
            elif mode == "mean":
                cond = feat.mean(dim=1)
            else:
                raise ValueError(f"Unknown mode {mode}")
            cond_list.append(cond)
        global_cond_seq = torch.stack(cond_list, dim=1)
        return global_cond_seq

    # -----------------------------
    # 3. 遍历数据
    # -----------------------------
    full_x, full_y = [], []

    with torch.no_grad():
        for i, batch in enumerate(train_dataloader):
            # device 转移
            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

            mod = self.ema_model
            nobs = mod.normalizer.normalize(batch['obs'])
            nactions = mod.normalizer['action'].normalize(batch['action'])

            # -----------------------------
            # 生成每步 global_cond_seq
            # -----------------------------
            global_cond_seq = build_sliding_window_cond(nobs, mod, mode="flatten")
            # global_cond_seq: (B, T - n_obs_steps + 1, ?)

            # -----------------------------
            # 对齐动作
            # -----------------------------
            trajectory = nactions.reshape(nactions.shape[0], -1)
            # 如果你想每步对应动作，可以裁掉前 K-1 步
            aligned_trajectory = trajectory[:, mod.n_obs_steps-1:, ...]

            print(f'At batch {i}/{len(train_dataloader)}')
            print(f'X: {global_cond_seq.shape}, Y: {aligned_trajectory.shape}')

            full_x.append(global_cond_seq.cpu())
            full_y.append(aligned_trajectory.cpu())

    # -----------------------------
    # 4. 拼接保存
    # -----------------------------
    full_x = torch.cat(full_x, dim=0)
    full_y = torch.cat(full_y, dim=0)
    print(f'Full X: {full_x.shape}, Full Y: {full_y.shape}')

    save_path = os.path.join(self.output_dir, 'full_data.pt')
    os.makedirs(self.output_dir, exist_ok=True)
    torch.save({'X': full_x, 'Y': full_y}, save_path)
    print(f'Full data saved to {save_path}')

                

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetHybridWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
