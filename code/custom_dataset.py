import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from pathlib import Path


PROJECT_PATH = Path(__file__).parents[1]
DATA_PATH = Path(PROJECT_PATH, "data")
NPY_VIDEO_PATH = Path(PROJECT_PATH, "data/processed_videos/IJA")
SPLIT_CSV_FILE = "ija_diagnosis_sets.csv"


class VideoDataset(Dataset):
    def __init__(
        self,
        group: str,
        split_csv_file=SPLIT_CSV_FILE,
    ):
        """
        Args:
            split_csv_file: "ija_diagnosis_sets.csv"
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        target_data = self.data.iloc[idx]
        target_path = get_video_path(target_data["file_name"])
        x_path = Path(NPY_VIDEO_PATH, f"{target_path.stem}.npy")
        return np.load(x_path), target_data["label"].item()


def get_video_path(file_name: str) -> Path:
    return Path(NPY_VIDEO_PATH, file_name)


def create_data_loader(batch_size):
    transform = transforms.ToTensor()
    train_loader = DataLoader(
        VideoDataset("train"),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    valid_loader = DataLoader(
        VideoDataset("valid"),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        VideoDataset("test"),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    train_loader, _, _ = create_data_loader(2)
    for batch_idx, (X, y) in enumerate(train_loader):
        print(X.shape)
        print(y)
        break
    # _, _, test_loader = create_data_loader(1)
    # for batch_idx, (X, y) in enumerate(test_loader):
    #     print(X.shape)
    #     print(y)
    #     break
