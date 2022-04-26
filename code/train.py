import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from custom_dataset import create_data_loader
from model import CNNLRCN
from earlystopping import EarlyStopping
from pathlib import Path
import logging


PROJECT_PATH = Path(__file__).parents[1]
CHECKPOINT_PATH = Path(PROJECT_PATH, "checkpoint/checkpoint.pt")

logger = logging.getLogger()
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(Path(PROJECT_PATH, "checkpoint/train_log.log"))
logger.addHandler(file_handler)

BATCH_SIZE = 2
N_EPOCHS = 6
PATIENCE = 3


def train_model(
    model,
    n_epochs,
    train_loader,
    valid_loader,
    optimizer,
    criterion,
    early_stopping,
    device,
    scheduler,
):
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []
    train_acc = []
    valid_acc = []
    logger.info(f"Training Start. Model name: {model.model_name}")

    for epoch in range(1, n_epochs + 1):
        # train the model#
        model.train()  # prep model for training
        for batch_idx, (X, y) in enumerate(train_loader, 1):
            X = X.float().to(device)
            y = y.to(device)

            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # record training loss
            train_losses.append(loss.item())

            pred = torch.argmax(output, dim=1)
            correct = (pred == y).sum().float()
            batch_acc = torch.round((correct / len(y) * 100).cpu())
            train_acc.append(batch_acc)
            print(
                f"epoch: {epoch}; batch:{batch_idx}/{len(train_loader)}; train_loss:{loss.item():.2f}"
            )
        scheduler.step()
        # validate the model#
        model.eval()  # prep model for evaluation
        for batch_idx, (X, y) in enumerate(valid_loader, 1):
            X = X.float().to(device)
            y = y.to(device)
            # forward pass: 입력된 값을 모델로 전달하여 예측 출력 계산
            output = model(X)
            # calculate the loss
            loss = criterion(output, y)

            pred = torch.argmax(output, dim=1)
            correct = (pred == y).sum().float()
            batch_acc = torch.round((correct / len(y) * 100).cpu())
            valid_acc.append(batch_acc)
            # record validation loss
            valid_losses.append(loss.item())
            print(
                f"epoch: {epoch}; batch:{batch_idx}/{len(valid_loader)}; valid_loss:{loss.item():.2f}"
            )

        # print 학습/검증 statistics
        # epoch당 평균 loss 계산
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        epoch_train_acc = np.average(train_acc)
        epoch_valid_acc = np.average(valid_acc)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(n_epochs))

        print_msg = (
            f"[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] "
            + f"train_loss: {train_loss:.5f} "
            + f"train_accuracy: {epoch_train_acc:.5f}"
            + f"valid_loss: {valid_loss:.5f}"
            + f"valid_accuracy: {epoch_valid_acc:.5f}"
        )

        logger.info(print_msg)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        # early_stopping는 validation loss가 감소하였는지 확인이 필요하며, 만약 감소하였을경우 현제 모델을 checkpoint로 만든다.
        early_stopping(valid_loss, model, epoch)

        if early_stopping.early_stop:
            logger.info("Early stopping")
            break

    # best model이 저장되어있는 last checkpoint를 로드한다.
    model.load_state_dict(torch.load(CHECKPOINT_PATH))

    return model, avg_train_losses, avg_valid_losses


def main():
    model = CNNLRCN(model_name="vgg16lrcn")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()  # binary에서는 BCELoss 사용하는게 좀 더 낫다는?
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=2, gamma=0.1
    )  # cLearning rate scheduler:
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True)

    train_loader, valid_loader, test_loader = create_data_loader(BATCH_SIZE)
    model, train_loss, valid_loss = train_model(
        model,
        N_EPOCHS,
        train_loader,
        valid_loader,
        optimizer,
        criterion,
        early_stopping,
        device,
        scheduler,
    )


if __name__ == "__main__":
    main()
