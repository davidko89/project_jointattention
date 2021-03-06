#%%
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from pathlib import Path
from tqdm import tqdm


SPLIT_CSV_FILE = "rja_low_videofile_with_dx.csv"
PROJECT_PATH = Path(__file__).parents[1]
DATA_PATH = Path(PROJECT_PATH, "data")
RAW_DATA_PATH = Path(DATA_PATH, "raw_data")
PROC_DATA_PATH = Path(DATA_PATH, "proc_data")
PROC_RJA_LOW_PATH = Path(DATA_PATH, "proc_data/proc_rja_low")


def read_dataset(csv_file: str) -> pd.DataFrame:
    dataset_path = Path(DATA_PATH, csv_file)
    return pd.read_csv(dataset_path)[["file_name", "label"]]


def get_video_path(task_name, file_name) -> Path:
    return Path(RAW_DATA_PATH, task_name, file_name)


def get_npy_output_path(file_name) -> Path:
    return Path(PROC_RJA_LOW_PATH, file_name)


def read_video(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype("uint8"))

    fc = 0
    ret = True

    while fc < frameCount and ret:
        ret, buf[fc] = cap.read()
        fc += 1

    cap.release()

    return np.transpose(buf, (0, 3, 2, 1))


def preproc_transform(video_arr):
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Grayscale(3),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            
        ]
    )
    return torch.cat(
        [
            preprocess(frame.transpose(2, 1, 0)).unsqueeze(0)
            for frame in pad_frame(video_arr)
        ],
        axis=0,
    )


def pad_frame(video_arr, target_length=150):
    length = video_arr.shape[0]
    if length >= target_length:
        new_video_arr = video_arr[:target_length]
    elif length < target_length:
        new_video_arr = pad_along_axis(video_arr, target_length)
    return new_video_arr


def pad_along_axis(array, target_length, axis=0):
    pad_size = target_length - array.shape[axis]

    if pad_size <= 0:
        return array

    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)

    return np.pad(array, pad_width=npad, mode="constant", constant_values=0)


def save_numpy_arr(arr: np.ndarray, file_name: str):
    np.save(get_npy_output_path(file_name), arr)


def process_by_file(task_name, file_name):
    output_file_name = f"{file_name.split('.')[0]}.npy"
    if Path.exists(get_npy_output_path(output_file_name)):
        return;

    target_path = get_video_path(task_name, file_name)
    video_arr = read_video(target_path)
    preproc_arr = preproc_transform(video_arr)

    # print(preproc_arr.shape)
    save_numpy_arr(preproc_arr, output_file_name)


#%%
def main():
    data: pd.DataFrame = read_dataset(SPLIT_CSV_FILE).dropna()
    task_name = "rja_low" 
    output_path = PROC_RJA_LOW_PATH
    output_files = [p.stem for p in output_path.glob("*.npy") if p]
    target_files = [f for f in data.file_name if f not in output_files]

    for idx, file_name in tqdm(enumerate(target_files)):
        # if idx == 5:
        #     break
        process_by_file(task_name, file_name)


if __name__ == "__main__":
    main()

    # for folder in PROC_DATA_PATH.glob("proc_rja_low"):
    #     for file in folder.glob("B015_RJA_low_1.npy"):
    #         arr = np.load(file)
    #         print(arr[150,].shape)
    #         plt.imshow(arr[150,].transpose(1, 2, 0))
    #         break
