#%%
import cv2 
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset 
from skimage.transform import resize
from pathlib import Path

PROJECT_PATH= Path(__file__).parents[1]
DATA_PATH = Path(PROJECT_PATH, 'data')
DATA_FILE_NAME ='ija_label_train.csv'
# task_folder = {'IJA', 'RJA_low', 'RJA_high_BL', 'RJA_high_BR', 'RJA_high_Lt', 'RJA_high_Rt'}

#%%
class VideoDataset(Dataset):
    """Video dataset."""

    def __init__(self, data_file_name:str, train: bool):
        """
        Args:
            train_data: using split_dataset, created new csv_file ('ija_trainset.csv')
            test_data: using split_dataset, created new csv_file ('ija_testset.csv')
            timesep: number of frames
            rgb: number of color channels
            h: height
            w: width
        """
        data = pd.read_csv(Path(DATA_PATH, data_file_name))
        
        if train:
            self.data = data.loc[data.train].reset_index(drop=True)
        elif not train:
            self.data = data.loc[~data.train].reset_index(drop=True)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        target_data = self.data.iloc[idx]
        target_path = get_video_path(target_data['filename'], "IJA")

        video = np.load()
        
        return torch.from_numpy(video), target_data['label'].item()
       


def read_dataset(file_name:str)->pd.DataFrame:
    full_path = Path(PROJECT_PATH, 'data', file_name)
    return pd.read_csv(full_path)[["file_name", "label"]]


def get_video_path(file_name, task)->Path:
    return Path(PROJECT_PATH, 'data/assembly', task, file_name)


def capture(file_path,timesep,rgb,h,w):
    resized_frames = np.zeros((timesep, rgb, 224, 224), dtype=np.float)
    
    vc = cv2.VideoCapture(str(file_path))

    for i in range(timesep):
        _, frame = vc.read()
        resized_frame = resize(frame,(224, 224, rgb))
        resized_frame = np.expand_dims(resized_frame,axis=0)
        if(np.max(resized_frame)>1):
            resized_frame = resized_frame/255.0
        resized_frame = np.moveaxis(resized_frame, -1, 1)
        resized_frames[i][:] = resized_frame # - tmp

    return resized_frames

# %%
if __name__ =='__main__':
    train_dataset = VideoDataset(DATA_FILE_NAME, True)
    x, y = train_dataset[1]
    print(x.shape)
# %%
