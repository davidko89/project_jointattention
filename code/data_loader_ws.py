import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.transforms import transforms
import numpy as np
import pandas as pd
from pathlib import Path


def get_split_csv_file_name(task, data_path):
    return Path(data_path, f"{task.name.lower()}_diagnosis_sets_1150.csv")


def get_cnn_path(task, data_path):
    return Path(data_path, f"proc_data/cnn_{task.name.lower()}")


# SPLIT_CSV_FILE = "rja_high_diagnosis_sets.csv" or "rja_low_diagnosis_sets.csv" or "ija_diagnosis_sets.csv"
# CNN_IJA_PATH = Path(DATA_PATH, "proc_data/cnn_ija")
# CNN_RJA_LOW_PATH = Path(DATA_PATH, "proc_data/cnn_rja_low")
# CNN_RJA_HIGH_PATH = Path(DATA_PATH, "proc_data/cnn_rja_high")


class VideoDataset(Dataset):
    def __init__(self, task, group: str, data_path, transform=None):
        """
        Args:
            group: str ("train", "valid", "test")
        """
        data = pd.read_csv(get_split_csv_file_name(task, data_path))

        # if group == "train":
        #     self.data = data.loc[data.train].reset_index(drop=True)
        # elif group == "valid":
        #     self.data = data.loc[data.valid].reset_index(drop=True)
        # elif group == "test":
        #     self.data = data.loc[data.test].reset_index(drop=True)
        # else:
        #     raise "group must be train, valid, or test"

        assert group in {"train", "valid", "test"}
        self.data = data.loc[data[group]].reset_index(drop=True)
        self.transform = transform
        self.cnn_data_path = get_cnn_path(task, data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        target_data = self.data.iloc[idx]
        target_path = Path(self.cnn_data_path, target_data["file_name"])
        X_path = Path(self.cnn_data_path, f"{target_path.stem}.npy")
        X = np.load(X_path)
        y = target_data["label"].item()
        return X, y


def get_loader(task, batch_size, data_path):
    my_transform = transforms.ToTensor()

    train_dataset = VideoDataset(task, "train", data_path, my_transform)
    valid_dataset = VideoDataset(task, "valid", data_path, my_transform)
    test_dataset = VideoDataset(task, "test", data_path, my_transform)

    class_weights = [1, 3]
    sample_weights = [0] * len(train_dataset)

    for batch_idx, (_, y) in enumerate(train_dataset):
        sample_weights[batch_idx] = class_weights[y]

    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    train_loader, _, _ = get_loader(batch_size=16)

    num_TD = 0
    num_ASD = 0

    for batch_idx, (X, y) in enumerate(train_loader):
        num_TD += torch.sum(y == 0)
        num_ASD += torch.sum(y == 1)
    print(num_TD)
    print(num_ASD)
