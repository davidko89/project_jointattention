#%%
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path
from tqdm import tqdm


PROJECT_PATH = Path(__file__).parents[1]
DATA_PATH = Path(PROJECT_PATH, "data")
PROC_IJA_PATH = Path(DATA_PATH, "proc_data/proc_ija")
PROC_RJA_LOW_PATH = Path(DATA_PATH, "proc_data/proc_rja_low")
PROC_RJA_HIGH_PATH = Path(DATA_PATH, "proc_data/proc_rja_high")
CNN_IJA_PATH = Path(DATA_PATH, "proc_data/cnn_ija")
CNN_IJA_BGR_PATH = Path(DATA_PATH, "proc_data/cnn_ija_bgr")
CNN_RJA_LOW_PATH = Path(DATA_PATH, "proc_data/cnn_rja_low")
CNN_RJA_HIGH_PATH = Path(DATA_PATH, "proc_data/cnn_rja_high")


SEQ_LEN = 300  # 150 if RJA_high, 300 if IJA or RJA_low


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


#%%
def main():
    model = CNN()
    device = torch.device("cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for file in tqdm(PROC_IJA_PATH.iterdir()):
        file_name = str.split(file.as_posix(), "/")[-1]
        output_file_path = Path(CNN_IJA_BGR_PATH, file_name)
        if Path.exists(output_file_path):
            continue

        arr = torch.Tensor(np.load(file)).to(device)
        # print(arr.shape)
        # if arr.shape[1] != 3:
        #     continue
        new_arr = model(arr)
        save_numpy_arr(new_arr.detach().numpy(), output_file_path)


if __name__ == "__main__":
    main()

    # for folder in PROC_DATA_PATH.glob("proc_ija"):
    #     for file in folder.glob("B015_IJA_1.npy"):
    #         arr = np.load(file)
    #         print(arr.shape)
