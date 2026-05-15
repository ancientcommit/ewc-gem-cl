from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


class MNISTVariantsSequentialDataModule:
    def __init__(
        self,
        data_dir="./data",
        batch_size=64,
        num_workers=2,
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = (
            transforms.ToTensor()
        )  # no normalization because we use different datasets

    def setup(self):
        """Setup task-specific datasets."""
        # yann.lecun.com is permanently down; patch mirrors before download
        torchvision.datasets.MNIST.mirrors = [
            "https://ossci-datasets.s3.amazonaws.com/mnist/",
            "https://storage.googleapis.com/cvdf-datasets/mnist/",
        ]
        self.train_dataset_a = torchvision.datasets.MNIST(
            root=self.data_dir, train=True, download=True, transform=self.transform
        )
        self.test_dataset_a = torchvision.datasets.MNIST(
            root=self.data_dir, train=False, download=True, transform=self.transform
        )

        self.train_dataset_b = torchvision.datasets.FashionMNIST(
            root=self.data_dir, train=True, download=True, transform=self.transform
        )
        self.test_dataset_b = torchvision.datasets.FashionMNIST(
            root=self.data_dir, train=False, download=True, transform=self.transform
        )

        # Maybe also add KMNIST later (website seems to be down)?

        # Print statistics
        print("\n--- TASK DATASETS ---")
        print(f"Task A (MNIST): {len(self.train_dataset_a)} train, {len(self.test_dataset_a)} test")
        print(f"Task B (FashionMNIST): {len(self.train_dataset_b)} train, {len(self.test_dataset_b)} test")

    def get_train_loader(self, task: str) -> DataLoader:
        """Get train loader for specific task."""
        dataset = getattr(self, f"train_dataset_{task.lower()}")
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def get_test_loader(self, task: str) -> DataLoader:
        """Get test loader for specific task."""
        dataset = getattr(self, f"test_dataset_{task.lower()}")
        return DataLoader(
            dataset,
            batch_size=256,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
