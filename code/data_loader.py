#%%
import pandas as pd
import torch
from torch.utils.data import Dataset 
from pathlib import Path

PROJECT_PATH= Path(__file__).parents[1]
DATA_PATH = Path(PROJECT_PATH, 'data')
VIDEO_PATH = Path(DATA_PATH, 'processed_videos')
SPLIT_CSV_FILE ='ija_label_train.csv'

#%%
class VideoDataset(Dataset):

    def __init__(self, train: bool, split_csv_file= SPLIT_CSV_FILE):
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
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        target_data = self.data.iloc[idx]
        target_path = get_video_path(target_data['file_name'])
        return torch.from_numpy(target_path), target_data['label'].item()
       

def get_video_path(file_name:str)->Path:
    return Path(VIDEO_PATH, 'IJA', file_name)

# %%
if __name__ =='__main__':
    train_dataset = VideoDataset(True)
    test_dataset = VideoDataset(False)

    x, y = train_dataset[0]
    print(x.shape)