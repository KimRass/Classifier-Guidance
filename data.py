from torchvision.datasets import CIFAR10
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np

transformer = T.Compose(
    [T.ToTensor(), T.Normalize(mean=0.5, std=0.5)],
)


def get_train_and_val_dls(data_dir, batch_size, n_cpus, seed, val_ratio=0.1):
    train_val_ds = CIFAR10(root=data_dir, train=True, download=True, transform=transformer)
    train_idx, val_idx = train_test_split(
        np.arange(len(train_val_ds)),
        test_size=val_ratio,
        random_state=seed,
        shuffle=True,
        stratify=[sample[1] for sample in train_val_ds],
    )
    train_ds = Subset(train_val_ds, train_idx)
    val_ds = Subset(train_val_ds, val_idx)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        num_workers=n_cpus,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        num_workers=n_cpus,
    )
    return train_dl, val_dl


def get_test_dl(data_dir, batch_size, n_cpus):
    test_ds = CIFAR10(root=data_dir, train=False, download=True, transform=transformer)
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=False,
        drop_last=True,
        persistent_workers=False,
        num_workers=n_cpus,
    )
    return test_dl
