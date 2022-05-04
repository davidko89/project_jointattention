import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path
from tqdm import tqdm


PROJECT_PATH = Path(__file__).parents[1]
DATA_PATH = Path(PROJECT_PATH, "data")
PROC_DATA_PATH = Path(DATA_PATH, "proc_data")
PROC_IJA_PATH = Path(DATA_PATH, "proc_data/proc_ija")
CNN_VIDEO_PATH = Path(DATA_PATH, "proc_data/cnn_ija")

SEQ_LEN = 300


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.basemodel = models.vgg16(pretrained=True).features

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        # X = X.view(X.size(0), -1)
        out = self.basemodel(x)
        return out
        # number of features = 512 * 7 * 7


def save_numpy_arr(arr: np.ndarray, file_path: Path):
    np.save(file_path, arr)


def main():
    model = CNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for file in tqdm((PROC_IJA_PATH).glob("*.npy")):
        arr = torch.Tensor(np.load(file)).to(device)
        print(arr.shape)
        if arr.shape[1] != 3:
            continue
        new_arr = model(arr)
        output_file_path = Path(CNN_VIDEO_PATH, file)
        save_numpy_arr(new_arr.detach().cpu().numpy(), output_file_path)


if __name__ == "__main__":
    main()
