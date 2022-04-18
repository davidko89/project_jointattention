#%%
import pandas as pd
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from pathlib import Path

PROJECT_PATH= Path(__file__).parents[1]
DATA_PATH = Path(PROJECT_PATH, 'data')
NPY_VIDEO_PATH = Path(PROJECT_PATH, 'data/processed_videos/IJA')
NPY_VIDEO_PATH.glob('*.npy')
SPLIT_CSV_FILE ='ija_label_train.csv'

#%%
class VideoDataset(Dataset):

    def __init__(self, train: bool, data_path, split_csv_file, transform=None):
        """
        Args:
            split_csv_file: 'ija_label_train.csv'
            timesep: number of frames
            train: True
        """
        with open(split_csv_file, 'rb') as f:
            self.label = pickle.load(f)

        self.data = np.load(data_path)

        
        self.transform = transform
    

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        target_label = self.label.iloc[idx]
        target_data = self.data[idx]
        
        return torch.Tensor(target_data), torch.Tensor(target_label['label'])
       

def get_video_path(file_name:str)->Path:
    return Path(NPY_VIDEO_PATH, file_name)

                                                                             
# %%
if __name__ =='__main__':
    train_dataset = VideoDataset(True)
    test_dataset = VideoDataset(False)

    from tqdm import tqdm
    lengths = {str(d[2]):d[0].shape[0] for d in tqdm(train_dataset) if d[0].shape[0] != 300}
    print(lengths)