from abc import ABC, abstractmethod
import torch.nn as nn


class ContinualLearningStrategy(ABC):
    """Abstract base class for continual learning strategies."""
    
    @abstractmethod
    def before_task(self, model: nn.Module, task_id: str, num_tasks: int) -> None:
        """Called before training on a new task.
        
        Args:
            model: The neural network model
            task_id: Identifier for the current task (e.g., 'A', 'B', 'C')
        """
        pass
    
    @abstractmethod
    def compute_loss(self, model: nn.Module, images, outputs, targets, task_id: str):
        """Compute loss with strategy-specific modifications.
        
        Args:
            model: The neural network model
            images: Model inputs
            outputs: Model predictions
            targets: Ground truth labels
            task_id: Identifier for the current task
            
        Returns:
            Loss tensor
        """
        pass

    def after_backward(self, model: nn.Module, task_id: str) -> None:
        """Optional hook called after backward pass.

        Args:
            model: The neural network model
            task_id: Identifier for the current task
        """
        pass
    
    @abstractmethod
    def after_task(self, model: nn.Module, task_id: str, train_loader) -> None:
        """Called after training on a task.
        
        Used for strategies that need to store information between tasks
        (e.g., store data for replay, compute Fisher information).
        
        Args:
            model: The neural network model
            task_id: Identifier for the completed task
            train_loader: DataLoader for the task's training data
        """
        pass
