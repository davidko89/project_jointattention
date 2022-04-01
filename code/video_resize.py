from pathlib import Path
import numpy as np
import pandas as pd
import cv2

PROJECT_PATH = Path(__file__).parents[1]
HEIGHT = 224
WEIGHT = 224

task_folder = {'1':'IJA', '2':'RJA_low', '3': 'RJA_high_BL', '4': 'RJA_high_BR', '5': 'RJA_high_Lt', '6': 'RJA_high_Rt'}

def read_dataset(file_name:str)->pd.DataFrame:
    full_path = Path(PROJECT_PATH, 'data', file_name)
    return pd.read_csv(full_path)[["file_name", "task", "label"]]

def get_video_path(file_name, task)->Path:
    return Path(PROJECT_PATH, 'data/assembly', task_folder[str(task)], file_name)

def resize_frame(frame_input, width=WEIGHT, height = HEIGHT): 
    return cv2.resize(frame_input, (width, height), interpolation=cv2.INTER_AREA)

def read_video(video_path:Path):
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        ret, frame = cap.read()
        resized_frame = resize_frame(frame)
        (h, w) = resized_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter('video_output.mp4', fourcc, 15.0, (w, h), True)

    while cap.isOpened():
        ret, frame = cap.read()
        resized_frame = resize_frame(frame)

        writer.write(resized_frame)

        cap.release()
       
def save_resized_video():
    pass

def main():
    data:pd.DataFrame = read_dataset("dataset_videos.csv")
    
    target_data:pd.Series = data.iloc[0]
    
    target_path = get_video_path(target_data['file_name'], target_data['task'])

    resized_video_path = resize_frame(target_path)

    resized_video = read_video(resized_video_path)
        
    print(resized_video.shape)

    # saved_resized_video = save_resized_video(resized_video) 

if __name__ =='__main__':
    main()