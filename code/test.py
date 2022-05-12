import numpy as np
import torch
import torch.nn as nn
from data_loader import get_loader
from lrcn_model import LRCN
from pathlib import Path
import logging
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
)


PROJECT_PATH = Path(__file__).parents[1]
CHECKPOINT_PATH = Path(
    PROJECT_PATH, "checkpoint/vgg16lrcn_weight_2.pt"
)  # specify which weight


logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s|%(levelname)s|%(message)s")
file_handler = logging.FileHandler(Path(PROJECT_PATH, "checkpoint/test_log.log"))
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_trained_network(model, batch_size, test_loader, criterion, device):
    # Initialize the prediction and label lists(tensors)
    predlist = torch.zeros(0, dtype=torch.long, device="cpu")
    lbllist = torch.zeros(0, dtype=torch.long, device="cpu")

    with torch.no_grad():
        train_loader, valid_loader, test_loader = get_loader(BATCH_SIZE)
        for i, (X, y) in enumerate(test_loader):
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            outputs = model(X)
            _, preds = torch.max(outputs, 1)
            # Append batch prediction results
            predlist = torch.cat([predlist, preds.view(-1).cpu()])
            lbllist = torch.cat([lbllist, y.view(-1).cpu()])

    # Confusion matrix
    conf_mat = confusion_matrix(lbllist.numpy(), predlist.numpy())
    print(conf_mat)
    logger.info(conf_mat)

    # Per-class accuracy
    class_accuracy = 100 * conf_mat.diagonal() / conf_mat.sum(1)
    print(class_accuracy)

    for f in [accuracy_score, precision_score, recall_score, roc_auc_score]:
        print(f(lbllist.numpy(), predlist.numpy()))
        logger.info(f(lbllist.numpy(), predlist.numpy()))


def main():
    model = LRCN(
        model_name="vgg16lrcn",
        dropout=0.4,
        seq_len=300,
        num_lstm_layers=1,
        lstm_hidden_dim=128,
    )
    device = DEVICE
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    train_loader, valid_loader, test_loader = get_loader(BATCH_SIZE)
    test_trained_network(
        model,
        BATCH_SIZE,
        test_loader,
        criterion,
        device,
    )


if __name__ == "__main__":
    main()
