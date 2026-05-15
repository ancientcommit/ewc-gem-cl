from .base import ContinualLearningStrategy
import torch.nn as nn
import torch.nn.functional as F


class NaiveStrategy(ContinualLearningStrategy):
    """Naive continual learning strategy - standard training without mitigation."""
    
    def before_task(self, model: nn.Module, task_id: str, num_tasks: int) -> None:
        """No preparation needed for naive strategy."""
        pass

    def compute_loss(self, model: nn.Module, images, outputs, targets, task_id: str):
        """Standard cross-entropy loss with no modifications."""
        return F.cross_entropy(outputs, targets)

    def after_backward(self, model: nn.Module, task_id: str) -> None:
        """No gradient post-processing needed for naive strategy."""
        pass
    
    def after_task(self, model: nn.Module, task_id: str, train_loader) -> None:
        """No post-task processing needed for naive strategy."""
        pass
