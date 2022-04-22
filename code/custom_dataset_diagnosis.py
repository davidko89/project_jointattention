#%%
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from pathlib import Path

PROJECT_PATH= Path(__file__).parents[1]
DATA_PATH = Path(PROJECT_PATH, 'data')
NPY_VIDEO_PATH = Path(PROJECT_PATH, 'data/processed_videos/IJA')
SPLIT_CSV_FILE ='ija_diagnosis_train.csv'

#%%
class VideoDataset(Dataset):

    def __init__(self, train:bool, split_csv_file=SPLIT_CSV_FILE, transform=None):
        """
        Args:
            split_csv_file: 'ija_label_train.csv'
            timesep: number of frames
            train: True
        """
        data = pd.read_csv(Path(DATA_PATH, split_csv_file))

        if train:
            self.data = data.loc[data.train].reset_index(drop=True)

        elif not train:
            self.data = data.loc[~data.train].reset_index(drop=True)

        self.transform = transform
    

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        target_data = self.data.iloc[idx]
        target_path = get_video_path(target_data['file_name'])
        x_path = Path(NPY_VIDEO_PATH,f"{target_path.stem}.npy")

        return np.load(x_path), target_data['label'].item()


def get_video_path(file_name:str)->Path:
    return Path(NPY_VIDEO_PATH, file_name)
                                              
# %%
if __name__ =='__main__':
    train_dataset = VideoDataset(True)
    test_dataset = VideoDataset(False)
    X, y = train_dataset[0]
    print(X.dtype)
    print(type(y))