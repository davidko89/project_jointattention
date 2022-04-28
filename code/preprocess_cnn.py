from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from tqdm import tqdm


PROJECT_PATH = Path(__file__).parents[1]
PROC_DATA_PATH = Path(PROJECT_PATH, "data/processed_videos")
CNN_VIDEO_PATH = Path(PROJECT_PATH, "data/processed_videos/IJA_CNN")

SEQ_LEN = 300


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.seq_len = SEQ_LEN
        self.basemodel = models.vgg16(pretrained=True).features

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        # seq_len, c, h, w = x.size()
        # reshape input to be (batch_size * seq_len, input_size)
        # x = x.view(seq_len, c, h, w)
        out = self.basemodel(x)
        return out
        # number of features = 512 * 7 * 7


def save_numpy_arr(arr: np.ndarray, file_name: str):
    print(Path(CNN_VIDEO_PATH, file_name))
    np.save(Path(CNN_VIDEO_PATH, file_name), arr)


def main():
    model = CNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    file_lists = []
    for folder in PROC_DATA_PATH.glob("IJA"):
        for file in folder.glob("*.npy"):
            file_lists.append(file)

        for file in tqdm(file_lists):
            arr = torch.Tensor(np.load(file)).to(device)
            new_arr = model(arr)
            save_numpy_arr(new_arr.detach().cpu().numpy(), f"{file}")

    # for file in~ probably brought path info as well thereby defeating the purpose of assigning a new folder for save

    # for file in folder.glob("D137_IJA_6.npy"):
    #     arr = torch.Tensor(np.load(file)).to(device)
    #     new_arr = model(arr)
    #     print(new_arr.shape)
    #     save_numpy_arr(new_arr.detach().cpu().numpy(), f"{file}")

    # for folder in PROC_DATA_PATH.glob("IJA"):
    #     for file in folder.glob("D705_IJA_4.npy"):
    #         arr = np.load(file)
    #         print(arr.shape)


if __name__ == "__main__":
    main()
