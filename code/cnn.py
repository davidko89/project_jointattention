import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from preprocess_dataset import get_loader
from pathlib import Path
from tqdm import tqdm


MNT_PATH = "/mnt/2021_NIA_data/jointattention/"
CNN_VIDEO_PATH = Path(MNT_PATH, "processed_videos/CNN_IJA")
SPLIT_CSV_FILE = "ija_diagnosis_sets.csv"

SEQ_LEN = 300


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.seq_len = SEQ_LEN
        self.basemodel = models.vgg16(pretrained=True).features

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, X):
        out = self.basemodel(X)
        return out
        # number of features = 512 * 7 * 7


def save_numpy_arr(arr: np.ndarray):
    np.save(Path(CNN_VIDEO_PATH), arr)


if __name__ == "__main__":
    data_loader = get_loader(batch_size=16)
    model = CNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for X, _ in tqdm(data_loader):
        X = X.float().to(device)
        output = model(X)
        save_numpy_arr(output)
