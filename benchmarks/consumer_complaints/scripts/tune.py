#!/usr/bin/env python3
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="config")
def tune(cfg: DictConfig) -> float:
    print(cfg.pretty())
    return 0


if __name__ == "__main__":
    tune()