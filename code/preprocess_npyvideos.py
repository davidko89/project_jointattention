#%%
from pathlib import Path
import numpy as np
from tqdm import tqdm

PROJECT_PATH = Path(__file__).parents[1]
NPY_VIDEO_PATH = Path(PROJECT_PATH, 'data/processed_videos')


def pad_along_axis(array, target_length, axis):
    pad_size = target_length - array.shape[axis]

    if pad_size <= 0:
        return array
    
    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)

    return np.pad(array, pad_width=npad, mode='constant', constant_values=0)


def pad_along_axis1(array, target_length):
    result = np.zeros(target_length*3*224*224).reshape(target_length, 3, 224, 224)
    result[:array.shape[0]] = array
    return result


def fix_to_same_length(video_arr):
    length = video_arr.shape[0]    

    if length >= 300:
        new_video_arr = video_arr[:300]
    elif length < 300:
        new_video_arr = pad_along_axis1(video_arr, 300)
    return new_video_arr

    # return video_arr[:300] if video_arr.shape[0] >= 300 else pad_along_axis1(video_arr, 300)

#%%
def save_numpy_arr(arr:np.ndarray, file_name:str):
    np.save(Path(NPY_VIDEO_PATH, file_name), arr)


def main():
    for folder in NPY_VIDEO_PATH.glob('IJA'):
        for file in tqdm(folder.glob('*.npy')):
            arr = np.load(file)
            new_arr = fix_to_same_length(arr)
            save_numpy_arr(new_arr, f"{file}")
            
    # for folder in NPY_VIDEO_PATH.glob('IJA'):
    #     for file in folder.glob('D137_IJA_6.npy'): 
    #         arr = np.load(file)
    #         new_arr = fix_to_same_length(arr)
    #         save_numpy_arr(new_arr, f"{file}")


if __name__ =='__main__':
    main()

# 'B014_IJA_1.npy' #length=300
# 'B014_IJA_5.npy' #length=158
# 'D137_IJA_6.npy' #length=330
# 'D703_IJA_5.npy' #length=306