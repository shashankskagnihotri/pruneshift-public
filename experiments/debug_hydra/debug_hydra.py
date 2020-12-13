import os

import hydra

@hydra.main(config_path="config", config_name="main")
def run(cfg):
    print(cfg)


if __name__ == "__main__":
    run()
