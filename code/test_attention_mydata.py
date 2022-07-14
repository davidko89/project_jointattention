from pathlib import Path
from enum import Enum, auto
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from data_loader import get_loader
from custom_model import CustomNet


class Task(Enum):
    IJA = auto()
    RJA_LOW = auto()
    RJA_HIGH = auto()


PROJECT_PATH = Path(__file__).parents[1]
DATA_PATH = Path(PROJECT_PATH, "data")
FIG_PATH = Path(PROJECT_PATH, "figures")
CHECKPOINT_PATH = Path(
    PROJECT_PATH, "checkpoint/lrcnatten_ija(bgr)_220714_weight_10.pt"
)
BATCH_SIZE = 1
SEQ_LEN = 300  # IJA: 300, RJA_high or RJA_low: 150


def test_trained_network(model, test_loader, device):

    model.eval()

    # Initialize label lists(tensors)
    predlist = torch.zeros(0, dtype=torch.long, device="cpu")
    lbllist = torch.zeros(0, dtype=torch.long, device="cpu")
    alphas_arrs = []

    for idx, (X, y) in enumerate(test_loader, 1):
        X = X.to(device)
        y = y.to(device)
        output, alphas_t = model(X)

        alphas_arr = alphas_t.detach().cpu().numpy()
        alphas_arrs.extend(alphas_arr)

        pred = torch.argmax(output, dim=1)

        # Append batch prediction results
        predlist = torch.cat([predlist, pred.view(-1).cpu()])
        lbllist = torch.cat([lbllist, y.view(-1).cpu()])

    conf_mat = confusion_matrix(lbllist.numpy(), predlist.numpy())
    print(conf_mat)
    return np.array(alphas_arrs), torch.LongTensor(lbllist).numpy()


def main(task: Task):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CustomNet(
        input_size=25088,
        seq_len=SEQ_LEN,
        num_hiddens=512,
        num_layers=2,
        dropout=0.5,
        attention_dim=SEQ_LEN,
    )

    model.to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH))

    _, _, test_loader = get_loader(task, BATCH_SIZE, DATA_PATH)

    alphas_arrs, labels = test_trained_network(
        model,
        test_loader,
        device,
    )

    fig, ax = plt.subplots()

    asd_alphas_arrs = alphas_arrs[labels == 1]
    td_alphas_arrs = alphas_arrs[labels == 0]

    one_asd_alphas_arr = (np.split(asd_alphas_arrs, 16))[15].reshape(-1)
    one_td_alphas_arr = (np.split(td_alphas_arrs, 41))[39].reshape(-1)

    plt.plot(one_asd_alphas_arr, c="r", label="ASD")
    plt.plot(one_td_alphas_arr, c="b", label="TD")
    plt.legend()
    plt.savefig(Path(FIG_PATH, f"ija_attention_visualization_bgr_39.png"))


if __name__ == "__main__":
    task = Task.IJA
    main(task)
