
import os
import os.path
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
from pathlib import Path
from typing import List, Union, Tuple, Any

PROJECT_PATH = Path(__file__).parents[1]
DATA_PATH = Path(PROJECT_PATH, "data")
RAW_DATA_PATH = Path(DATA_PATH, "raw_data")

class VideoRecord(object):
    """
    Helper class for class VideoFrameDataset
    Args:
        root_datapath: system path to root folder of videos
        row: 1) first element is path to video frames excluding
             the root_datapath prefix 
             2) second element is starting frame id of video
             3) third element is inclusive ending frame id of video
             4) fourth element is label index.
    """
    def __init__(self, row, root_datapath):
        self._data = row
        self._path = os.path.join(root_datapath, row[0])

    @property
    def path(self) -> str:
        return self._path

    @property
    def num_frames(self) -> int:
        return self.end_frame - self.start_frame + 1  # +1 because end frame is inclusive
    @property
    def start_frame(self) -> int:
        return int(self._data[1])

    @property
    def end_frame(self) -> int:
        return int(self._data[2])

    @property
    def label(self) -> Union[int, List[int]]:
        # just one label_id
        if len(self._data) == 4:
            return int(self._data[3])


class VideoFrameDataset(torch.utils.data.Dataset):
    """Args:
        root_path: 
            root path in which video folders lie. 
        annotationfile_path: 
            .txt annotation file containing one row per video sample 
        num_segments: 
            number of segments the video should be divided into to sample frames from.
        frames_per_segment: 
            The number of frames that should be loaded per segment. For each segment's frame-range, a random start index or center is chosen, from which frames_per_segment consecutive frames are loaded.
        imagefile_template: 
            image filename template that video frame files have inside of their video folders
        transform: 
            Transform pipeline that receives a list of PIL images/frames.
        test_mode: 
            If True, frames are taken from the center of each segment, instead of random location in each segment."""

def __init__(self,
                 root_path: str,
                 annotationfile_path: str,
                 num_segments: int = 3,
                 frames_per_segment: int = 1,
                 imagefile_template: str='img_{:05d}.jpg',
                 transform = None,
                 test_mode: bool = False):
        super(VideoFrameDataset, self).__init__()

        self.root_path = root_path
        self.annotationfile_path = annotationfile_path
        self.num_segments = num_segments
        self.frames_per_segment = frames_per_segment
        self.imagefile_template = imagefile_template
        self.transform = transform
        self.test_mode = test_mode

        self._parse_annotationfile()
        self._sanity_check_samples()

def _load_image(self, directory: str, idx: int) -> Image.Image:
    return Image.open(os.path.join(directory, self.imagefile_template.format(idx))).convert('RGB')

def _parse_annotationfile(self):
    self.video_list = [VideoRecord(x.strip().split(), self.root_path) for x in open(self.annotationfile_path)]

def _sanity_check_samples(self):
    for record in self.video_list:
        if record.num_frames <= 0 or record.start_frame == record.end_frame:
            print(f"\nDataset Warning: video {record.path} seems to have zero RGB frames on disk!\n")

        elif record.num_frames < (self.num_segments * self.frames_per_segment):
            print(f"\nDataset Warning: video {record.path} has {record.num_frames} frames "
                    f"but the dataloader is set up to load "
                    f"(num_segments={self.num_segments})*(frames_per_segment={self.frames_per_segment})"
                    f"={self.num_segments * self.frames_per_segment} frames. Dataloader will throw an "
                    f"error when trying to load this video.\n")

def _get_start_indices(self, record: VideoRecord) -> 'np.ndarray[int]':
    """
    For each segment, choose a start index from where frames
    are to be loaded from.
    Args:
        record: VideoRecord denoting a video sample.
    Returns:
        List of indices of where the frames of each
        segment are to be loaded from.
    """
    # choose start indices that are perfectly evenly spread across the video frames.
    if self.test_mode:
        distance_between_indices = (record.num_frames - self.frames_per_segment + 1) / float(self.num_segments)

        start_indices = np.array([int(distance_between_indices / 2.0 + distance_between_indices * x)
                                    for x in range(self.num_segments)])
    # randomly sample start indices that are approximately evenly spread across the video frames.
    else:
        max_valid_start_index = (record.num_frames - self.frames_per_segment + 1) // self.num_segments

        start_indices = np.multiply(list(range(self.num_segments)), max_valid_start_index) + \
                    np.random.randint(max_valid_start_index, size=self.num_segments)

    return start_indices

def __getitem__(self, idx: int) -> Union[
    Tuple[List[Image.Image], Union[int, List[int]]],
    Tuple['torch.Tensor[num_frames, channels, height, width]', Union[int, List[int]]],
    Tuple[Any, Union[int, List[int]]],
    ]:
    """
    For video with id idx, loads self.NUM_SEGMENTS * self.FRAMES_PER_SEGMENT
    frames from evenly chosen locations across the video.
    Args:
        idx: Video sample index.
    Returns:
        A tuple of (video, label). Label is either a single
        integer or a list of integers in the case of multiple labels.
        Video is either 1) a list of PIL images if no transform is used
        2) a batch of shape (NUM_IMAGES x CHANNELS x HEIGHT x WIDTH) in the range [0,1]
        if the transform "ImglistToTensor" is used
        3) or anything else if a custom transform is used.
    """
    record: VideoRecord = self.video_list[idx]

    frame_start_indices: 'np.ndarray[int]' = self._get_start_indices(record)

    return self._get(record, frame_start_indices)

def _get(self, record: VideoRecord, frame_start_indices: 'np.ndarray[int]') -> Union[
    Tuple[List[Image.Image], Union[int, List[int]]],
    Tuple['torch.Tensor[num_frames, channels, height, width]', Union[int, List[int]]],
    Tuple[Any, Union[int, List[int]]],
    ]:
    """
    Loads the frames of a video at the corresponding
    indices.
    Args:
        record: VideoRecord denoting a video sample.
        frame_start_indices: Indices from which to load consecutive frames from.
    Returns:
        A tuple of (video, label). Label is either a single
        integer or a list of integers in the case of multiple labels.
        Video is either 1) a list of PIL images if no transform is used
        2) a batch of shape (NUM_IMAGES x CHANNELS x HEIGHT x WIDTH) in the range [0,1]
        if the transform "ImglistToTensor" is used
        3) or anything else if a custom transform is used.
    """

    frame_start_indices = frame_start_indices + record.start_frame
    images = list()

    # from each start_index, load self.frames_per_segment
    # consecutive frames
    for start_index in frame_start_indices:
        frame_index = int(start_index)

        # load self.frames_per_segment consecutive frames
        for _ in range(self.frames_per_segment):
            image = self._load_image(record.path, frame_index)
            images.append(image)

            if frame_index < record.end_frame:
                frame_index += 1

    if self.transform is not None:
        images = self.transform(images)

    return images, record.label

def __len__(self):
    return len(self.video_list)

class ImglistToTensor(torch.nn.Module):
    """
    Converts a list of PIL images in the range [0,255] to a torch.FloatTensor
    of shape (NUM_IMAGES x CHANNELS x HEIGHT x WIDTH) in the range [0,1].
    Can be used as first transform for ``VideoFrameDataset``.
    """
    @staticmethod
    def forward(img_list: List[Image.Image]) -> 'torch.Tensor[NUM_IMAGES, CHANNELS, HEIGHT, WIDTH]':
        """
        Converts each PIL image in a list to
        a torch Tensor and stacks them into
        a single tensor.
        Args:
            img_list: list of PIL images.
        Returns:
            tensor of size ``NUM_IMAGES x CHANNELS x HEIGHT x WIDTH``
        """
        return torch.stack([transforms.functional.to_tensor(pic) for pic in img_list])