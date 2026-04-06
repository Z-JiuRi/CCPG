import argparse
from omegaconf import OmegaConf

from core.trainer import Trainer
from core.inferencer import Inferencer

def main():
    parser = argparse.ArgumentParser(description='LoRA Diffusion Training')
    parser.add_argument('--config', type=str, default='', help="")
    parser.add_argument('--mode', type=str, default='', help="")
    args = parser.parse_args()
    
    cfg = OmegaConf.load(args.config)

    if args.mode == 'train':
        trainer = Trainer(cfg)
        trainer.train()
    else:
        inferencer = Inferencer(cfg)
        inferencer.inference()

if __name__ == '__main__':
    main()