#%%
from pathlib import Path
from enum import Enum, auto
import logging
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
import torch
import matplotlib.pyplot as plt
from data_loader import get_loader
from custom_model import CustomNet


class Task(Enum):
    IJA = auto()
    RJA_LOW = auto()
    RJA_HIGH = auto()


PROJECT_PATH = Path(__file__).parents[1]
DATA_PATH = Path(PROJECT_PATH, "data")
CHECKPOINT_PATH = Path(
    PROJECT_PATH, "checkpoint/lrcn_atten_ija_ws_220701_weight_10.pt"
)  # specify which weight
BATCH_SIZE = 1
SEQ_LEN = 300  # 300 if IJA & RJA_low, 150 if RJA_high


logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s|%(levelname)s|%(message)s")
file_handler = logging.FileHandler(Path(PROJECT_PATH, "checkpoint/test_log.log"))
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def test_trained_network(model, test_loader, device):
    # Initialize the prediction and label lists(tensors)
    predlist = torch.zeros(0, dtype=torch.long, device="cpu")
    lbllist = torch.zeros(0, dtype=torch.long, device="cpu")

    model.eval()

    for batch_idx, (X, y) in enumerate(test_loader, 1):
        X = X.to(device)
        y = y.to(device)

        output, alphas_t = model(X)
        
        alphas_arr = alphas_t.detach().cpu().numpy()

        pred = torch.argmax(output, dim=1)

        # Append batch prediction results
        predlist = torch.cat([predlist, pred.view(-1).cpu()])
        lbllist = torch.cat([lbllist, y.view(-1).cpu()])

    conf_mat = confusion_matrix(lbllist.numpy(), predlist.numpy())
    
    logger.info(f"confusion_matrix: {conf_mat}")

    acc_score = accuracy_score(lbllist.numpy(), predlist.numpy())
    logger.info(f"accuracy_score: {acc_score}")

    prec_score = precision_score(lbllist.numpy(), predlist.numpy())
    logger.info(f"precision_score: {prec_score}")

    rec_score = recall_score(lbllist.numpy(), predlist.numpy())
    logger.info(f"recall_score: {rec_score}")

    roc_score = roc_auc_score(lbllist.numpy(), predlist.numpy())
    logger.info(f"roc_auc_score: {roc_score}")

    f1 = f1_score(lbllist.numpy(), predlist.numpy())
    logger.info(f"f1_score: {f1}")

    return alphas_arr


def main(task: Task):
    logger.info(f"{task.name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    model = CustomNet(
        input_size=25088,
        seq_len=SEQ_LEN,
        num_hiddens=128,
        num_layers=2,
        dropout=0.4,
        attention_dim=128,
    )

    model.to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH))

    _, _, test_loader = get_loader(task, BATCH_SIZE, DATA_PATH)

    alphas_arr = test_trained_network(
        model,
        test_loader,
        device,  
    )

    fig, ax = plt.subplots()
    im = ax.imshow(alphas_arr.squeeze(0).reshape(-1, 300))
    plt.plot(alphas_arr.squeeze(0))


if __name__ == "__main__":
    task = Task.IJA
    main(task)

    # model = CustomNet(
    #     input_size=25088,
    #     seq_len=150,
    #     num_hiddens=128,
    #     num_layers=2,
    #     dropout=0.4,
    #     attention_dim=128,)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    # model.load_state_dict(torch.load(CHECKPOINT_PATH))
    # X = torch.rand(size=(1, 150, 512, 7, 7))
    # X = X.to(device)
    # output, alphas_t = model(X)
    # alphas_arr = alphas_t.detach().cpu().numpy()
    # fig, ax = plt.subplots()
    # im = ax.imshow(alphas_arr.squeeze(0).reshape(-1, 150))
    # plt.plot(alphas_arr.squeeze(0))

# %%
