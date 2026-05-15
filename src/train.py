import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from pathlib import Path
import sys
import warnings

# Add src to path (so that we can import with paths from project root)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics.clmetrics import CLMetrics

# Suppress torchvision image extension load warnings (native libjpeg missing)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"torchvision\.io\.image"
)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print("="*60)
    print("Catastrophic Forgetting Experiment")
    print("="*60)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Set seed
    torch.manual_seed(cfg.experiment.seed)
    
    # Device setup
    if cfg.experiment.trainer.device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 
                             ('mps' if torch.backends.mps.is_available() else 'cpu'))
    else:
        device = torch.device(cfg.experiment.trainer.device)
    print(f"\nUsing device: {device}")
    
    # Instantiate datamodule
    datamodule = hydra.utils.instantiate(cfg.experiment.dataset)
    datamodule.setup()
    
    # Instantiate model
    model = hydra.utils.instantiate(cfg.experiment.model).to(device)
    print(f"\nModel:\n{model}")

    # Instantiate strategy
    strategy = hydra.utils.instantiate(cfg.experiment.strategy)
    
    # Instantiate optimizer
    optimizer = hydra.utils.instantiate(cfg.experiment.optimizer, params=model.parameters())
    
    # Instantiate metrics tracker
    metrics = CLMetrics()
    
    # Instantiate catastrophic forgetting trainer
    trainer = hydra.utils.instantiate(cfg.experiment.trainer, strategy=strategy, model=model, optimizer=optimizer, device=device, metrics=metrics)
    
    # Train through all tasks
    trainer.fit(datamodule, tasks=cfg.experiment.task_order)

if __name__ == "__main__":
    main()