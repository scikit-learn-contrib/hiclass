import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="../configs", config_name="logistic_regression")
def optimize(cfg : DictConfig) -> float:
    classifier__solver: str = cfg.classifier__solver
    classifier__max_iter: int = cfg.classifier__max_iter
    return 1

if __name__ == "__main__":
    optimize()