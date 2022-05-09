import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.transforms import transforms
import numpy as np
import pandas as pd
from pathlib import Path


SPLIT_CSV_FILE = "ija_diagnosis_sets.csv"
PROJECT_PATH = Path(__file__).parents[1]
DATA_PATH = Path(PROJECT_PATH, "data")
CNN_IJA_PATH = Path(DATA_PATH, "proc_data/cnn_ija")


class VideoDataset(Dataset):
    def __init__(self, group: str, split_csv_file, transform=None):
        """
        Args:
            group: str ("train", "valid", "test")
        """
        data = pd.read_csv(Path(DATA_PATH, split_csv_file))

        if group == "train":
            self.data = data.loc[data.train].reset_index(drop=True)
        elif group == "valid":
            self.data = data.loc[data.valid].reset_index(drop=True)
        elif group == "test":
            self.data = data.loc[data.test].reset_index(drop=True)
        else:
            raise "group must be train, valid, or test"

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        target_data = self.data.iloc[idx]
        target_path = get_video_path(target_data["file_name"])
        X_path = Path(CNN_IJA_PATH, f"{target_path.stem}.npy")
        X = np.load(X_path)
        y = target_data["label"].item()
        return X, y


def get_video_path(file_name: str) -> Path:
    return Path(CNN_IJA_PATH, file_name)


def get_loader(batch_size):
    my_transform = transforms.ToTensor()
    train_dataset = VideoDataset("train", SPLIT_CSV_FILE, my_transform)
    valid_dataset = VideoDataset("valid", SPLIT_CSV_FILE, my_transform)
    test_dataset = VideoDataset("test", SPLIT_CSV_FILE, my_transform)
    class_weights = [1, 6]
    sample_weights = [0] * len(train_dataset)

    for batch_idx, (X, y) in enumerate(train_dataset):
        class_weight = class_weights[y]
        sample_weights[batch_idx] = class_weight

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
