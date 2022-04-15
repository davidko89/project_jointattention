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

        return np.load(x_path), target_data['label'].item()
       

def get_video_path(file_name:str)->Path:
    return Path(VIDEO_PATH, 'IJA_npy', file_name)


# custom collate function
# def get_max_length(x):
#     return len(max(x, key=len))

# def pad_sequence(seq):
#     def _pad(_it, _max_len):
#         return [0] * (_max_len - len(_it)) + _it
#     return [_pad(it, get_max_length(seq)) for it in seq]

# def custom_collate(batch):
#     transposed = zip(*batch)
#     lst = []
#     for samples in transposed:
#         if isinstance(samples[0], int):
#             lst.append(torch.LongTensor(samples))
#         elif isinstance(samples[0], float):
#             lst.append(torch.DoubleTensor(samples))
#         elif isinstance(samples[0], collections.Sequence):
#             lst.append(torch.LongTensor(pad_sequence(samples)))
#     return lst

# stream_dataset = StreamDataset(data_path)
# stream_data_loader = torch.utils.data.dataloader.DataLoader(dataset=stream_dataset,                                                         
#                                                             batch_size=batch_size,                                            
#                                                         collate_fn=custom_collate,
#                                                         shuffle=False)

# %%
if __name__ =='__main__':
    train_dataset = VideoDataset(True)
    test_dataset = VideoDataset(False)
    lengths = []
    from tqdm import tqdm
    for i in tqdm(range(len(train_dataset))):
        length_ = train_dataset[i][0].shape[0]
        if length_ != 300:
            lengths.append(length_)

    print(lengths)

# %%
