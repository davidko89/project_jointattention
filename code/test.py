import numpy as np
import torch
import torch.nn as nn
from data_loader import create_data_loader
from lrcn_model import LRCN
from pathlib import Path
import logging

PROJECT_PATH = Path(__file__).parents[1]
CHECKPOINT_PATH = Path(PROJECT_PATH, "checkpoint/_.pt")  ##_specify

logger = logging.getLogger()
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(Path(PROJECT_PATH, "checkpoint/test_log.log"))
logger.addHandler(file_handler)


BATCH_SIZE = 16


def test_trained_network(model, batch_size, test_loader, criterion, device):
    # initialize lists to monitor test loss and accuracy
    test_loss = 0.0
    class_correct = list(0.0 for i in range(2))
    class_total = list(0.0 for i in range(2))

    model.eval()  # prep model for evaluation

    for data, target in test_loader:
        data = data.float().to(device)
        target = target.to(device)

        if len(target.data) != batch_size:
            break

        # forward pass
        output = model(data)
        # calculate loss
        loss = criterion(output, target)
        # update test loss
        test_loss += loss.item() * data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        # calculate test accuracy for each object class
        for i in range(batch_size):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # calculate and print avg test loss
    test_loss = test_loss / len(test_loader.dataset)
    logger.info("Test Loss: {:.6f}\n".format(test_loss))

    for i in range(2):
        if class_total[i] > 0:
            logger.info(
                "Test Accuracy of %5s: %2d%% (%2d/%2d)"
                % (
                    str(i),
                    100 * class_correct[i] / class_total[i],
                    np.sum(class_correct[i]),
                    np.sum(class_total[i]),
                )
            )
        else:
            logger.info(
                "Test Accuracy of %5s: N/A (no training examples)" % (class_total[i])
            )

    logger.info(
        "\nTest Accuracy (Overall): %2d%% (%2d/%2d)"
        % (
            100.0 * np.sum(class_correct) / np.sum(class_total),
            np.sum(class_correct),
            np.sum(class_total),
        )
    )


def main():
    model = LRCN(model_name="vgg16lrcn")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    train_loader, valid_loader, test_loader = create_data_loader(BATCH_SIZE)
    test_trained_network(
        model,
        BATCH_SIZE,
        test_loader,
        criterion,
        device,
    )


if __name__ == "__main__":
    main()
