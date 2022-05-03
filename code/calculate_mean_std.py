from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

SPLIT_CSV_FILE = "ija_videofile_with_dx.csv"

MNT_PATH = "/mnt/2021_NIA_data/jointattention/"
PROC_IJA_PATH = Path(MNT_PATH, "processed_videos/PROC_IJA")
CNN_VIDEO_PATH = Path(MNT_PATH, "processed_videos/CNN_IJA")


def get_video_path(file_name: str) -> Path:
    return Path(PROC_IJA_PATH, file_name)

def get_mean_std(X):
    mean = torch.mean(X, dim=[0, 2, 3])
    std = torch.std(X, dim=[0, 2, 3])
    return mean, std

def main():
    X_path_iterator = Path(PROC_IJA_PATH).glob("*.npy")
    for target_path in X_path_iterator:
        get_mean_std(torch.tensor(np.load(target_path)))


if __name__ == "__main__":
    main()
