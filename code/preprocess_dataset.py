from pathlib import Path
import numpy as np
import pandas as pd
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


SPLIT_CSV_FILE = "ija_videofile_with_dx.csv"

MNT_PATH = "/mnt/2021_NIA_data/jointattention/"
PROC_DATA_PATH = Path(MNT_PATH, "processed_videos")
PROC_IJA_PATH = Path(PROC_DATA_PATH, "PROC_IJA")
CNN_VIDEO_PATH = Path(PROC_DATA_PATH, "CNN_IJA")


class ProcDataset(Dataset):
    def __init__(self, split_csv_file, transform=None):
        self.data = pd.read_csv(Path(MNT_PATH, split_csv_file))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        target_data = self.data.iloc[idx]
        target_path = get_video_path(target_data["file_name"])
        X_path = Path(PROC_IJA_PATH, f"{target_path.stem}.npy")
        X = np.load(X_path)
        y = target_data["label"].item()
        return X, y


def get_video_path(file_name) -> Path:
    return Path(PROC_IJA_PATH, file_name)


def get_loader(batch_size):
    my_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.CenterCrop((448, 448)),
            transforms.Resize((224, 224)),
            transforms.Normalize((0, 0, 0), (0, 0, 0)),
        ]
    )
    proc_dataset = ProcDataset(SPLIT_CSV_FILE, my_transforms)
    data_loader = DataLoader(proc_dataset, batch_size=batch_size, num_workers=16)
    return data_loader


if __name__ == "__main__":
    data_loader = get_loader(batch_size=2)
    for i, (X, y) in data_loader:
        print(X)
        break
