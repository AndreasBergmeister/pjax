import gzip
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import requests
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision.datasets import CIFAR10, MNIST, VisionDataset
from tqdm import tqdm


class DataModule(ABC):
    """Base class for dataset loading and preprocessing."""

    def __init__(
        self,
        batch_size: int = 64,
        normalize: bool = True,
        data_dir: str = "./dataset",
        overfit_batches: int = 0,
        preload: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.normalize = normalize
        self.data_dir = data_dir
        self.preload = preload
        self.seed = seed

        # Load train/val dataset
        full_dataset = self.dataset_cls(data_dir, train=True, download=True)
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [0.9, 0.1], generator=torch.Generator().manual_seed(self.seed)
        )
        if overfit_batches:
            self.train_dataset = Subset(self.train_dataset, range(overfit_batches * batch_size))

        # Load test dataset
        self.test_dataset = self.dataset_cls(data_dir, train=False, download=True)

        if self.preload:
            print("Preloading data...")

            def _preload(dataset, name):
                return [item for item in tqdm(dataset, desc=f"Preloading {name}")]

            self.train_dataset = _preload(self.train_dataset, "train")
            self.val_dataset = _preload(self.val_dataset, "validation")
            self.test_dataset = _preload(self.test_dataset, "test")
            print("Data preloaded.")

    @property
    @abstractmethod
    def dataset_cls(self) -> type[VisionDataset]:
        pass

    @property
    @abstractmethod
    def data_mean(self):
        pass

    @property
    @abstractmethod
    def data_std(self):
        pass

    def collate_fn(self, batch):
        images, labels = zip(*batch)
        x = np.array(images, dtype=np.float32) / 255.0
        y = np.array(labels, dtype=np.int64)
        if x.ndim == 3:
            x = x[..., None]  # add channel dim
        if self.normalize:
            x = (x - np.array(self.data_mean)) / np.array(self.data_std)
        return x, y

    def train_iterator(self):
        train_loader = self.train_dataloader()

        def _iterator():
            while True:
                for batch in train_loader:
                    yield batch

        return iter(_iterator())

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
            collate_fn=self.collate_fn,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            drop_last=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=False,
            pin_memory=True,
        )


class MNISTDataModule(DataModule):
    """MNIST dataset module with standard preprocessing."""

    @property
    def dataset_cls(self):
        return MNIST

    @property
    def data_mean(self):
        return (0.1307,)

    @property
    def data_std(self):
        return (0.3081,)


class CIFAR10DataModule(DataModule):
    """CIFAR-10 dataset module with standard preprocessing."""

    @property
    def dataset_cls(self):
        return CIFAR10

    @property
    def data_mean(self):
        return (0.4914, 0.4822, 0.4465)

    @property
    def data_std(self):
        return (0.2023, 0.1994, 0.2010)


class CIFAR10_CModule(CIFAR10DataModule):
    """Corrupted CIFAR10 dataset (only affects the test set)"""

    class CorruptedCIFAR10(Dataset):
        def __init__(self, corruptions, labels):
            self.corruptions = corruptions
            self.labels = labels

        def __len__(self):
            return len(self.corruptions) * len(self.labels)

        def __getitem__(self, idx):
            corruption_idx = idx // len(self.labels)
            corruption_key = sorted(self.corruptions.keys())[corruption_idx]
            label_idx = idx % len(self.labels)
            return self.corruptions[corruption_key][label_idx], self.labels[label_idx]

    def __init__(
        self,
        batch_size: int = 64,
        normalize: bool = True,
        data_dir: str = "./dataset",
        preload: bool = False,
        seed: int = 42,
    ):
        # Initialize train/val using parent class, test will be overwritten below
        super().__init__(batch_size, normalize, data_dir=data_dir, preload=preload, seed=seed)

        # download the corrupted CIFAR10 test set if it doesn't exist
        import tarfile
        from pathlib import Path

        import requests

        url = "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar"
        path = Path(data_dir) / "CIFAR-10-C"
        path.mkdir(parents=True, exist_ok=True)
        file = path / "CIFAR-10-C.tar"
        if not file.exists():
            with requests.get(url, stream=True) as r:
                with open(file, "wb") as f:
                    f.write(r.content)

            with tarfile.open(file) as f:
                f.extractall(path.parent)

        # load files
        corruptions = {}
        labels = None
        for file in path.glob("*.npy"):
            name = file.stem
            if name == "labels":
                labels = np.load(file)
            else:
                corruptions[name] = np.load(file)

        self.test_dataset = self.CorruptedCIFAR10(corruptions, labels)


class HIGGSDataModule:
    """HIGGS particle physics dataset module."""

    class HiggsDataset(Dataset):
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def __len__(self):
            return len(self.x)

        def __getitem__(self, idx):
            return self.x[idx], self.y[idx]

    def __init__(
        self,
        batch_size: int = 64,
        normalize: bool = True,
        data_dir: str = "./dataset",
        preload: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.normalize = normalize
        self.data_dir = data_dir
        self.seed = seed

        # download the Higgs dataset if it doesn't exist
        path = Path(data_dir) / "HIGGS"
        path.mkdir(parents=True, exist_ok=True)
        csv_file = path / "HIGGS.csv"
        if not csv_file.exists():
            print(f"{csv_file} does not exist. Obtaining it...")
            higgs_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
            gz_file = path / "HIGGS.csv.gz"
            if not gz_file.exists():
                print(f"Downloading {higgs_url}...")
                with requests.get(higgs_url, stream=True) as r:
                    r.raise_for_status()
                    with open(gz_file, "wb") as f:
                        shutil.copyfileobj(r.raw, f)
                print("Download complete.")
            else:
                print(f"{gz_file} already exists. Skipping download.")

            print(f"Extracting {gz_file} to {csv_file}...")
            with gzip.open(gz_file, "rb") as f_in:
                with open(csv_file, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print("Extraction complete.")

        # load the dataset
        data = np.loadtxt(csv_file, delimiter=",")
        x = data[:, 1:]
        y = data[:, 0].astype(np.int64)

        # compute mean and std
        self.data_mean = x.mean(axis=0)
        self.data_std = x.std(axis=0)

        # split the dataset
        # last 500k samples are used for testing
        x_test, y_test = x[-500_000:], y[-500_000:]
        x_train_val, y_train_val = x[:-500_000], y[:-500_000]

        split = int(0.9 * len(x_train_val))
        shuffle = np.random.default_rng(self.seed).permutation(len(x_train_val))

        x_train, y_train = x_train_val[shuffle[:split]], y_train_val[shuffle[:split]]
        x_val, y_val = x_train_val[shuffle[split:]], y_train_val[shuffle[split:]]

        self.train_dataset = x_train, y_train
        self.val_dataset = x_val, y_val
        self.test_dataset = x_test, y_test

    def collate_fn(self, batch):
        x, y = zip(*batch)
        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.int64)
        if self.normalize:
            x = (x - self.data_mean) / self.data_std
        return x, y

    def train_iterator(self):
        train_loader = self.train_dataloader()

        def _iterator():
            while True:
                for batch in train_loader:
                    yield batch

        return iter(_iterator())

    def train_dataloader(self):
        return DataLoader(
            self.HiggsDataset(*self.train_dataset),
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
            collate_fn=self.collate_fn,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.HiggsDataset(*self.val_dataset),
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            drop_last=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.HiggsDataset(*self.test_dataset),
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=False,
            pin_memory=True,
        )


class ShakespeareDataModule:
    """Shakespeare text dataset module for character-level language modeling."""

    class TextDataset(Dataset):
        def __init__(self, text, seq_len):
            self.text = text
            self.seq_len = seq_len

        def __len__(self):
            return len(self.text) - self.seq_len

        def __getitem__(self, idx):
            return self.text[idx : idx + self.seq_len], self.text[idx + 1 : idx + self.seq_len + 1]

    def __init__(
        self,
        batch_size: int = 64,
        seq_len: int = 64,
        data_dir: str = "./dataset",
        preload: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.data_dir = data_dir
        self.seed = seed

        # download the Shakespeare dataset if it doesn't exist
        txt_file = Path(data_dir) / "shakespeare.txt"
        if not txt_file.exists():
            text_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            with requests.get(text_url, stream=True) as r:
                with open(txt_file, "wb") as f:
                    f.write(r.content)

        # load the dataset
        with open(txt_file, "r") as f:
            text = f.read()

        # create vocabulary
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        char_to_idx = {ch: i for i, ch in enumerate(chars)}

        # convert text to tensor
        text = np.array([char_to_idx[ch] for ch in text], dtype=np.int64)

        # last 20% samples are used for testing
        n = len(text)
        test_start = int(0.8 * n)
        train_end = int(0.9 * test_start)
        self.train_dataset = text[:train_end]
        self.val_dataset = text[train_end:test_start]
        self.test_dataset = text[test_start:]

    def collate_fn(self, batch):
        x, y = zip(*batch)
        return np.array(x, dtype=np.int64), np.array(y, dtype=np.int64)

    def train_iterator(self):
        train_loader = self.train_dataloader()

        def _iterator():
            while True:
                for batch in train_loader:
                    yield batch

        return iter(_iterator())

    def train_dataloader(self):
        return DataLoader(
            self.TextDataset(self.train_dataset, self.seq_len),
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
            collate_fn=self.collate_fn,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.TextDataset(self.val_dataset, self.seq_len),
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            drop_last=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.TextDataset(self.test_dataset, self.seq_len),
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=False,
            pin_memory=True,
        )
