import hydra
from omegaconf import OmegaConf
from diffusion_policy.dataset.robomimic_replay_image_dataset_for_libero import RobomimicReplayImageDataset

cfg_path = "/home/zhiyuanjia/FAIL-Detect/diffusion_policy/configs_robomimic"
cfg_name = "image_square_ph_visual_diffusion_policy_cnn_for_libero.yaml"

@hydra.main(config_path=cfg_path, config_name=cfg_name, version_base=None)
def test(cfg):
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    print(dataset)

test()