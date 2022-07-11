import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from custom_model_2 import CustomNet

# PROJECT_PATH = Path(__file__).parents[1]
# DATA_PATH = Path(PROJECT_PATH, "data")
# CHECKPOINT_PATH = Path(PROJECT_PATH, "checkpoint")
N_EPOCHS = 20
BATCH_SIZE = 200
SEQ_LEN = 10
FEATURE_LEN = 25088


def get_loader(batch_size):
    early_arr = np.ones(BATCH_SIZE * SEQ_LEN // 2 * FEATURE_LEN).reshape(
        BATCH_SIZE, SEQ_LEN // 2, -1
    )

    late_arr1 = np.ones(BATCH_SIZE * SEQ_LEN // 2 * FEATURE_LEN).reshape(
        BATCH_SIZE, SEQ_LEN // 2, -1
    )
    late_arr2 = np.zeros(BATCH_SIZE * SEQ_LEN // 2 * FEATURE_LEN).reshape(
        BATCH_SIZE, SEQ_LEN // 2, -1
    )

    label1 = np.concatenate([early_arr, late_arr1], axis=1) + np.random.normal(
        0.05, size=(BATCH_SIZE, SEQ_LEN, FEATURE_LEN)
    )
    label2 = np.concatenate([early_arr, late_arr2], axis=1) + np.random.normal(
        0.05, size=(BATCH_SIZE, SEQ_LEN, FEATURE_LEN)
    )
    X = torch.from_numpy(np.concatenate([label1, label2], axis=0)).float()
    y = torch.LongTensor(np.array([1.0] * BATCH_SIZE + [0.0] * BATCH_SIZE).reshape(-1))

    train_dataset = TensorDataset(X, y)
    # print(X.shape)
    # print(y.shape)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader


# model4 = CustomNet(
#     input_size=FEATURE_LEN,
#     seq_len=SEQ_LEN,
#     num_hiddens=128,
#     num_layers=2,
#     dropout=0.5,d
# )

# print(X.shape) for _ in range(100):
#     pred_y = model4(X)


def train_model(
    model,
    n_epochs,
    train_loader,
    optimizer,
    criterion,
    device,
):
    # initialize lists to monitor train, valid loss and accuracy
    train_losses = []
    avg_train_losses = []
    train_acc = []
    

    for epoch in range(1, n_epochs + 1):
        # train the model
        model.train()  # prep model for training
        labels = []
        alphas_arrs = []
        for batch_idx, (X, y) in enumerate(train_loader, 1):
            labels.extend(y)
            X = X.to(device)
            y = y.to(device)
            

            # forward pass
            output, alphas_t = model(X)
            alphas_arr = alphas_t.detach().cpu().numpy()
            alphas_arrs.extend(alphas_arr)
            
            loss = criterion(output, y)
            loss.backward()

            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # perform a single optimization step (parameter update)
            optimizer.step()

            # record training loss
            train_losses.append(loss.item())
            pred = torch.argmax(output, dim=1)
            correct = (pred == y).sum().float()
            batch_acc = torch.round((correct / len(y) * 100).cpu())
            train_acc.append(batch_acc)

        train_loss = np.average(train_losses)
        epoch_train_acc = np.average(train_acc)
        avg_train_losses.append(train_loss)

        epoch_len = len(str(n_epochs))

        print_msg = (
            f"[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] "
            + f"train_loss: {train_loss:.5f} "
            + f"train_accuracy: {epoch_train_acc:.5f}"
        )
        print(print_msg)

        # clear lists to track next epoch
        train_losses = []

        # torch.save(
        #     model.state_dict(), Path(CHECKPOINT_PATH, f"test_attention_weight.pt")
        # )

    return model, avg_train_losses, np.array(alphas_arrs), torch.Tensor(labels).numpy()


def main():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    model = CustomNet(
        input_size=25088,
        seq_len=SEQ_LEN,
        num_hiddens=128,
        num_layers=2,
        dropout=0.4,
        attention_dim=128,
    )

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    train_loader = get_loader(BATCH_SIZE)

    model, train_loss, alphas_arr, labels = train_model(
        model,
        N_EPOCHS,
        train_loader,
        optimizer,
        criterion,
        device,
    )
    fig, ax = plt.subplots()
    
    true_alphas_arr = alphas_arr[labels == 1].mean(0)
    false_alphas_arr = alphas_arr[labels == 0].mean(0)
    
    plt.plot(true_alphas_arr, c = 'b', label = 'True')
    plt.plot(false_alphas_arr, c = 'r', label = 'False')
    plt.legend()

    # im = ax.imshow(alphas_arr.reshape(-1, 10))

    # alphas_arr2 = alphas_arr.transpose(1,0)
    
    # plt.plot(alphas_arr.transpose(1,0))
    plt.savefig('Sample_Attention.png')


if __name__ == "__main__":
    main()
