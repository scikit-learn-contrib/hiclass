import hydra
from omegaconf import DictConfig, OmegaConf


def optimize_lightgbm(cfg: DictConfig) -> float:
    raise NotImplementedError("lightgbm")


@hydra.main(config_path="../configs", config_name="logistic_regression")
def optimize(cfg: DictConfig) -> float:
    classifier: str = cfg.classifier
    if classifier == "lightgbm":
        return optimize_lightgbm(cfg)
    # raise NotImplementedError(cfg)
    # classifier__solver: str = cfg.classifier__solver
    # classifier__max_iter: int = cfg.classifier__max_iter
    return 1


if __name__ == "__main__":
    optimize()
