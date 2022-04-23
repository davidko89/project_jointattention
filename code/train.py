from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from custom_dataset import create_data_loader
from model import VGG16LRCN
from pathlib import Path
from earlystopping import EarlyStopping


model = VGG16LRCN()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)


def train_model(model, batch_size, patience, n_epochs):
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []

    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in tqdm(range(1, n_epochs + 1)):
        # train the model#
        model.train()  # prep model for training
        for batch_idx, (X, y) in enumerate(train_loader):
            X = X.float().to(device)
            y = y.int().to(device)

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

        # validate the model#
        model.eval()  # prep model for evaluation
        for X, y in valid_loader:
            # forward pass: 입력된 값을 모델로 전달하여 예측 출력 계산
            output = model(X)
            # calculate the loss
            loss = criterion(output, y)
            # record validation loss
            valid_losses.append(loss.item())

        # print 학습/검증 statistics
        # epoch당 평균 loss 계산
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(n_epochs))

        print_msg = (
            f"[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] "
            + f"train_loss: {train_loss:.5f} "
            + f"valid_loss: {valid_loss:.5f}"
        )

        print(print_msg)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        # early_stopping는 validation loss가 감소하였는지 확인이 필요하며, 만약 감소하였을경우 현제 모델을 checkpoint로 만든다.
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # best model이 저장되어있는 last checkpoint를 로드한다.
    model.load_state_dict(torch.load("checkpoint.pt"))

    return model, avg_train_losses, avg_valid_losses


batch_size = 4
n_epochs = 10
patience = 7 # early stopping patience: validation loss가 개선된 마지막 시간 이후로 얼마나 기다릴지 지정

train_loader, test_loader, valid_loader = create_data_loader(batch_size)
model, train_loss, valid_loss = train_model(model, batch_size, patience, n_epochs)


if __name__ == "__main__":
    train_model()

    # import matplotlib.pyplot as plt
    # # 훈련이 진행되는 과정에 따라 loss를 시각화
    # fig = plt.figure(figsize=(10, 8))
    # plt.plot(range(1, len(train_loss) + 1), train_loss, label="Training Loss")
    # plt.plot(range(1, len(valid_loss) + 1), valid_loss, label="Validation Loss")
    # # validation loss의 최저값 지점을 찾기
    # minposs = valid_loss.index(min(valid_loss)) + 1
    # plt.axvline(minposs, linestyle="--", color="r", label="Early Stopping Checkpoint")
    # plt.xlabel("epochs")
    # plt.ylabel("loss")
    # plt.ylim(0, 0.5)  # 일정한 scale
    # plt.xlim(0, len(train_loss) + 1)  # 일정한 scale
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    # fig.savefig("loss_plot.png", bbox_inches="tight")