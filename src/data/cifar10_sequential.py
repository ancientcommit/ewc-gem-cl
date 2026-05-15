import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from typing import Dict, List


class CIFAR10SequentialDataModule:
    """DataModule for sequential task learning on CIFAR10."""
    
    def __init__(
        self,
        data_dir: str = './data',
        batch_size: int = 128,
        num_workers: int = 4,
        normalize_mean: List[float] = None,
        normalize_std: List[float] = None,
        tasks: Dict[str, List[int]] = None,
    ):
        """Initialize CIFAR10 DataModule.
        
        Args:
            data_dir: Directory to store/load data
            batch_size: Batch size for training
            num_workers: Number of workers for data loading
            normalize_mean: Normalization mean values
            normalize_std: Normalization std values
            tasks: Dictionary mapping task IDs to class indices
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Default CIFAR10 normalization for stability and faster training (parameters do not have to be changed as much)
        if normalize_mean is None:
            normalize_mean = [0.4914, 0.4822, 0.4465]
        if normalize_std is None:
            normalize_std = [0.2470, 0.2435, 0.2616]
        
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        
        # Default tasks: 3 tasks with 3/4 classes each
        if tasks is None:
            tasks = {
                'A': [0, 1, 2, 3],
                'B': [4, 5, 6],
                'C': [7, 8, 9]
            }
        self.tasks = tasks
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.normalize_mean, self.normalize_std)
        ])
        
        self.train_dataset = None
        self.test_dataset = None
        self.train_subsets = {}
        self.test_subsets = {}
    
    def setup(self):
        """Download and setup datasets."""
        self.train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.transform
        )
        self.test_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.transform
        )
        
        self._create_task_subsets()
    
    def _create_task_subsets(self):
        """Create train and test subsets for each task."""
        train_targets = torch.tensor(self.train_dataset.targets)
        test_targets = torch.tensor(self.test_dataset.targets)
        
        print("\n--- CIFAR10 TASK SUBSETS ---")
        for task_id, class_indices in self.tasks.items():
            # Create training subset
            train_mask = torch.zeros_like(train_targets, dtype=torch.bool)
            for class_idx in class_indices:
                train_mask |= (train_targets == class_idx)
            self.train_subsets[task_id] = Subset(
                self.train_dataset,
                train_mask.nonzero(as_tuple=True)[0].tolist()
            )
            
            # Create test subset
            test_mask = torch.zeros_like(test_targets, dtype=torch.bool)
            for class_idx in class_indices:
                test_mask |= (test_targets == class_idx)
            self.test_subsets[task_id] = Subset(
                self.test_dataset,
                test_mask.nonzero(as_tuple=True)[0].tolist()
            )
            
            print(
                f"Task {task_id} (classes {class_indices[0]}-{class_indices[-1]}): "
                f"{len(self.train_subsets[task_id])} train, {len(self.test_subsets[task_id])} test"
            )
    
    def get_train_loader(self, task_id: str) -> DataLoader:
        """Get training DataLoader for a specific task.
        
        Args:
            task_id: Task identifier (e.g., 'A', 'B')
            
        Returns:
            DataLoader for the task's training data
        """
        return DataLoader(
            self.train_subsets[task_id],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
    
    def get_test_loader(self, task_id: str) -> DataLoader:
        """Get test DataLoader for a specific task.
        
        Args:
            task_id: Task identifier (e.g., 'A', 'B')
            
        Returns:
            DataLoader for the task's test data
        """
        return DataLoader(
            self.test_subsets[task_id],
            batch_size=256,
            shuffle=False,
            num_workers=self.num_workers
        )
