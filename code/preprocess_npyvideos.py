from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import skimage.transform
from tqdm import tqdm


PROJECT_PATH = Path(__file__).parents[1]
PROC_DATA_PATH = Path(PROJECT_PATH, "data/processed_videos")
PROC_IJA_DATA_PATH = Path(PROJECT_PATH, "data/processed_videos/IJA")
RESIZED_HEIGHT, RESIZED_WIDTH = 224, 224
RGB = 3


def read_dataset(csv_file: str) -> pd.DataFrame:
    dataset_path = Path(PROJECT_PATH, "data", csv_file)
    return pd.read_csv(dataset_path)[["file_name", "label"]]


def get_video_path(task_name, file_name) -> Path:
    return Path(PROJECT_PATH, "data/assembly", task_name, file_name)


def read_video(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # buf = np.empty((frameCount, 3, frameWidth, frameHeight), np.dtype('uint8'))
    buf = np.empty((frameCount, 3, 224, 224), np.dtype(np.float32))

    for fc in range(frameCount):
        ret, frame_arr = cap.read()
        if not ret:
            continue
        buf[fc] = resize_video(frame_arr.transpose(2, 0, 1))
        break

    cap.release()
    return buf  # (N, C, H, W)


def resize_video(frame_arr):
    resized_frame = skimage.transform.resize(frame_arr, (224, 224, 3))
    resized_frame = np.expand_dims(resized_frame, axis=0)
    if np.max(resized_frame) > 1:
        resized_frame = resized_frame / 255.0
    resized_frame = np.moveaxis(resized_frame, -1, 1)
    return resized_frame


def pad_along_axis1(array, target_length):
    result = np.zeros(target_length * 3 * 224 * 224).reshape(target_length, 3, 224, 224)
    result[: array.shape[0]] = array
    return result


def fix_to_same_length(video_arr):
    length = video_arr.shape[0]

    if length >= 300:
        new_video_arr = video_arr[:300]
    elif length < 300:
        new_video_arr = pad_along_axis1(video_arr, 300)
    return new_video_arr

    # return video_arr[:300] if video_arr.shape[0] >= 300 else pad_along_axis1(video_arr, 300)


def save_numpy_arr(arr: np.ndarray, file_name: str):
    np.save(Path(PROC_IJA_DATA_PATH, file_name), arr)


def process_by_file(task_name, file_name):
    target_path = get_video_path(task_name, file_name)
    video_arr = read_video(target_path)
    new_arr = fix_to_same_length(video_arr)
    save_numpy_arr(new_arr, f"{file_name.split('.')[0]}.npy")


def main():
    data: pd.DataFrame = read_dataset("ija_videofile_with_dx.csv").dropna()

    task_name = "IJA"

    for file_name in tqdm(data.file_name):
        process_by_file(task_name, file_name)

    # for folder in PROC_DATA_PATH.glob("IJA"):
    #     for file in folder.glob("D137_IJA_6.npy"):
    # arr = np.load(file)
    # print(arr.shape)


if __name__ == "__main__":
    main()
