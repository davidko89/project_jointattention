from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import skimage.transform

PROJECT_PATH = Path(__file__).parents[1]
RESIZED_HEIGHT, RESIZED_WIDTH = 224, 224
RGB = 3
task_folder = {'IJA', 'RJA_low', 'RJA_high_BL', 'RJA_high_BR', 'RJA_high_Lt', 'RJA_high_Rt'}

def read_dataset(file_name:str)->pd.DataFrame:
    full_path = Path(PROJECT_PATH, 'data', file_name)
    return pd.read_csv(full_path)[["file_name", "label"]]

def get_video_path(file_name, task)->Path:
    return Path(PROJECT_PATH, 'data/assembly', task, file_name)

def read_video(video_path:Path)->np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # buf = np.empty((frameCount, 3, frameHeight, frameWidth), np.dtype('uint8'))
    buf = np.empty((frameCount, 3, 224, 224), np.dtype('uint8'))

    fc = 0
    ret = True

    while (fc < frameCount and ret):
        ret, frame_arr = cap.read()
        buf[fc] = resize_video(frame_arr.transpose(2, 0, 1))
        fc += 1

    cap.release()
    return buf # (N, C, H, W)

def resize_video(frame_arr):
    resized_frame = skimage.transform.resize(frame_arr,(224, 224, 3))
    resized_frame = np.expand_dims(resized_frame,axis=0)
    if(np.max(resized_frame)>1):
        resized_frame = resized_frame/255.0
    resized_frame = np.moveaxis(resized_frame, -1, 1)
    return resized_frame

def save_numpy_arr(arr:np.ndarray, file_name:str):
    np.save(Path(PROJECT_PATH, 'data/video_npy', file_name), arr)

def main():
    data: pd.DataFrame = read_dataset("ija_video_file_with_label.csv").dropna()
    
    for filename in tqdm(data.file_name):
        target_path = get_video_path(filename, 'IJA')
        video_arr = read_video(target_path)
        save_numpy_arr(video_arr, f"{filename.split('.')[0]}.npy")

if __name__ =='__main__':
    main()