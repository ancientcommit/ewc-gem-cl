import numpy as np
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from PIL import Image


class MNISTPermutationsSequentialDataModule:
    def __init__(
        self,
        data_dir="./data",
        batch_size=64,
        num_workers=2,
        normalize_mean=0.1307,
        normalize_std=0.3081,
        tasks=None,
    ):
        if tasks is None:
            tasks = ["A", "B", "C", "D", "E"]
        self.tasks = tasks
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((normalize_mean,), (normalize_std,)),
            ]
        )

    def setup(self):
        """Setup datasets and create task-specific subsets."""
        # yann.lecun.com is permanently down; patch mirrors before download
        torchvision.datasets.MNIST.mirrors = [
            "https://ossci-datasets.s3.amazonaws.com/mnist/",
            "https://storage.googleapis.com/cvdf-datasets/mnist/",
        ]

        # Create subsets for each task
        self._create_permutations()

    def _create_permutations(self):
        """Create train and test permutations for each task."""

        mnist_train = torchvision.datasets.MNIST(
            root=self.data_dir, train=True, download=True, transform=None
        )
        mnist_test = torchvision.datasets.MNIST(
            root=self.data_dir, train=False, download=True, transform=None
        )

        print("\n--- TASK PERMUTATIONS ---")
        for task in self.tasks:
            perm = np.random.permutation(28 * 28)

            train_perm_dataset = PermutedMNIST(mnist_train, perm, self.transform)
            test_perm_dataset = PermutedMNIST(mnist_test, perm, self.transform)

            setattr(self, f"train_perm_{task.lower()}", train_perm_dataset)
            setattr(self, f"test_perm_{task.lower()}", test_perm_dataset)

            # Print statistics
            print(
                f"Task {task}: {len(train_perm_dataset)} train, {len(test_perm_dataset)} test"
            )

    def get_train_loader(self, task: str) -> DataLoader:
        """Get train loader for specific task."""
        perm = getattr(self, f"train_perm_{task.lower()}")
        return DataLoader(
            perm,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def get_test_loader(self, task: str) -> DataLoader:
        """Get test loader for specific task."""
        perm = getattr(self, f"test_perm_{task.lower()}")
        return DataLoader(
            perm,
            batch_size=256,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )


class PermutedMNIST:
    """Wrapper that applies a pixel permutation to MNIST images."""

    def __init__(self, mnist_dataset, permutation, transform):
        """
        Args:
            mnist_dataset: MNIST dataset
            permutation: Pixel indices for permutation
            transform: Transforms to apply after permutation
        """
        self.mnist_dataset = mnist_dataset
        self.permutation = permutation
        self.transform = transform

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, index):
        image, label = self.mnist_dataset[index]
        perm_image = np.array(image).flatten()[self.permutation].reshape(28, 28)
        perm_image = Image.fromarray(perm_image.astype(np.uint8))

        if self.transform:
            perm_image = self.transform(perm_image)

        return perm_image, label
