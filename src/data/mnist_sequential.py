import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from typing import List

class MNISTSequentialDataModule:
    def __init__(
        self, 
        data_dir='./data', 
        batch_size=64, 
        num_workers=2,
        normalize_mean=0.1307, 
        normalize_std=0.3081,
        tasks=None
    ):
        if tasks is None:
            tasks = {
                'A': [0, 1, 2, 3],
                'B': [4, 5, 6],
                'C': [7, 8, 9]
            }
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.task_a_labels = tasks['A']
        self.task_b_labels = tasks['B']
        self.task_c_labels = tasks['C']
        
        # normalization for stability and faster training (parameters do not have to be changed as much)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((normalize_mean,), (normalize_std,))
        ])
        
    def setup(self):
        """Setup datasets and create task-specific subsets."""
        # yann.lecun.com is permanently down; patch mirrors before download
        torchvision.datasets.MNIST.mirrors = [
            "https://ossci-datasets.s3.amazonaws.com/mnist/",
            "https://storage.googleapis.com/cvdf-datasets/mnist/",
        ]
        self.train_dataset = torchvision.datasets.MNIST(
            root=self.data_dir, 
            train=True, 
            download=True, 
            transform=self.transform
        )
        self.test_dataset = torchvision.datasets.MNIST(
            root=self.data_dir, 
            train=False, 
            download=True, 
            transform=self.transform
        )
        
        # Create subsets for each task
        self._create_task_subsets()
        
    def _create_task_subsets(self):
        """Create train and test subsets for each task."""
        # Training subsets
        train_targets = self.train_dataset.targets
        
        mask_a = self._create_mask(train_targets, self.task_a_labels)
        mask_b = self._create_mask(train_targets, self.task_b_labels)
        mask_c = self._create_mask(train_targets, self.task_c_labels)
        
        self.train_subset_a = Subset(self.train_dataset, mask_a.nonzero(as_tuple=True)[0].tolist())
        self.train_subset_b = Subset(self.train_dataset, mask_b.nonzero(as_tuple=True)[0].tolist())
        self.train_subset_c = Subset(self.train_dataset, mask_c.nonzero(as_tuple=True)[0].tolist())
        
        # Test subsets
        test_targets = self.test_dataset.targets
        
        mask_a = self._create_mask(test_targets, self.task_a_labels)
        mask_b = self._create_mask(test_targets, self.task_b_labels)
        mask_c = self._create_mask(test_targets, self.task_c_labels)
        
        self.test_subset_a = Subset(self.test_dataset, mask_a.nonzero(as_tuple=True)[0].tolist())
        self.test_subset_b = Subset(self.test_dataset, mask_b.nonzero(as_tuple=True)[0].tolist())
        self.test_subset_c = Subset(self.test_dataset, mask_c.nonzero(as_tuple=True)[0].tolist())
        
        # Print statistics
        print("\n--- TASK SUBSETS ---")
        print(f"Task A (labels {self.task_a_labels}): {len(self.train_subset_a)} train, {len(self.test_subset_a)} test")
        print(f"Task B (labels {self.task_b_labels}): {len(self.train_subset_b)} train, {len(self.test_subset_b)} test")
        print(f"Task C (labels {self.task_c_labels}): {len(self.train_subset_c)} train, {len(self.test_subset_c)} test")
        
    def _create_mask(self, targets: torch.Tensor, labels: List[int]) -> torch.Tensor:
        """Create a boolean mask for the given labels."""
        mask = torch.zeros_like(targets, dtype=torch.bool)
        for label in labels:
            mask |= (targets == label)
        return mask
        
    def get_train_loader(self, task: str) -> DataLoader:
        """Get train loader for specific task."""
        subset = getattr(self, f'train_subset_{task.lower()}')
        return DataLoader(
            subset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
        
    def get_test_loader(self, task: str) -> DataLoader:
        """Get test loader for specific task."""
        subset = getattr(self, f'test_subset_{task.lower()}')
        return DataLoader(
            subset, 
            batch_size=256, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )