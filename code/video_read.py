from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from numpy import asarray
from numpy import save

PROJECT_PATH = Path(__file__).parents[1]

task_folder = {'1':'IJA', '2':'RJA_low', '3': 'RJA_high_BL', '4': 'RJA_high_BR', '5': 'RJA_high_Lt', '6': 'RJA_high_Rt'}

def read_dataset(file_name:str)->pd.DataFrame:
    full_path = Path(PROJECT_PATH, 'data', file_name)
    return pd.read_csv(full_path)[["file_name", "task", "label"]]

def get_video_path(file_name, task)->Path:
    return Path(PROJECT_PATH, 'data/assembly', task_folder[str(task)], file_name)

def read_video(video_path:Path)->np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

    fc = 0
    ret = True

    while (fc < frameCount  and ret):
        ret, buf[fc] = cap.read()
        fc += 1

    cap.release()
    return buf.transpose(0, 3, 1, 2) # (N, C, H, W)

def save_numpy_arr(arr:np.ndarray):
    video_arr_npy = asarray([])
    save('data.npy', video_arr_npy)

def main():
    data:pd.DataFrame = read_dataset("dataset_videos.csv")

    # target_data:pd.Series = data.iloc[0]
    for target_data in range(0, len(data)):
        
        target_path = get_video_path(target_data['file_name'], target_data['task'])

        video_arr = read_video(target_path)
        
        video_arr_npy = save_numpy_arr(video_arr)

        print(video_arr_npy.shape)
        print(video_arr_npy[0])

if __name__ =='__main__':
    main()