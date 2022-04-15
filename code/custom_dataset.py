#%%
import pandas as pd
import numpy as np
import torch 
from torch.utils.data import Dataset
from pathlib import Path

PROJECT_PATH= Path(__file__).parents[1]
DATA_PATH = Path(PROJECT_PATH, 'data')
VIDEO_PATH = Path(DATA_PATH, 'processed_videos')
SPLIT_CSV_FILE ='ija_label_train.csv'

#%%
class VideoDataset(Dataset):

    def __init__(self, train: bool, split_csv_file= SPLIT_CSV_FILE, transform=None):
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
        x_path = Path(PROJECT_PATH, "data/processed_videos/IJA_npy",f"{target_path.stem}.npy")

        return np.load(x_path), target_data['label'].item(), x_path
       

def get_video_path(file_name:str)->Path:
    return Path(VIDEO_PATH, 'IJA_npy', file_name)

                                                                                                               
# %%
if __name__ =='__main__':
    train_dataset = VideoDataset(True)
    test_dataset = VideoDataset(False)
    from tqdm import tqdm
    lengths = {str(d[2]):d[0].shape[0] for d in tqdm(train_dataset) if d[0].shape[0] != 300}

    print(lengths)
    
    # import numpy as np
    # data = np.arange()

    # for i in tqdm(range(len(train_dataset))):
    #         length_ = train_dataset[i][0].shape[0]
    #         if length_ > 300:
    #            numpy.delete(arr, obj = (300 - length), axis=None)
    #         if length_ < 300:
    #             numpy.pad(array, pad_width, mode='constatnt', **kwargs)