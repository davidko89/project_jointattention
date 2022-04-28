from pathlib import Path
import numpy as np
import cv2
import skimage.transform

PROJECT_PATH = Path(__file__).parents[1]
PROC_DATA_PATH = Path(PROJECT_PATH, "data/processed_videos")
PROC_IJA_DATA_PATH = Path(PROJECT_PATH, "data/processed_videos/IJA")


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


def main():
    video_path = Path(PROJECT_PATH, "data/assembly/IJA", "B015_IJA_2.mp4")
    a = read_video(video_path)
    
    print(a.mean(axis=(0, 2, 3)))
    print(a.std(axis=(0, 2, 3)))


if __name__ == "__main__":
    main()
