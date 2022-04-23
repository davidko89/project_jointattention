from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import skimage.transform
from tqdm import tqdm

PROJECT_PATH = Path(__file__).parents[1]
NPY_VIDEO_PATH = Path(PROJECT_PATH, "data/processed_videos/IJA")
RESIZED_HEIGHT, RESIZED_WIDTH = 224, 224
RGB = 3
# task_name = {'IJA', 'RJA_low', 'RJA_high_BL', 'RJA_high_BR', 'RJA_high_Lt', 'RJA_high_Rt'}


def read_dataset(csv_file: str) -> pd.DataFrame:
    dataset_path = Path(PROJECT_PATH, "data", csv_file)
    return pd.read_csv(dataset_path)[["file_name", "label"]]


def get_video_path(task_name: str, file_name) -> Path:
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


def save_numpy_arr(arr: np.ndarray, file_name: str):
    np.save(Path(NPY_VIDEO_PATH, file_name), arr)


def process_by_file(task_name, file_name):
    target_path = get_video_path(task_name, file_name)
    video_arr = read_video(target_path)
    save_numpy_arr(video_arr, f"{file_name.split('.')[0]}.npy")


def main():
    data: pd.DataFrame = read_dataset("ija_videofile_with_dx.csv").dropna()

    task_name = "IJA"

    for file_name in tqdm(data.file_name):
        process_by_file(file_name, task_name)


if __name__ == "__main__":
    main()
