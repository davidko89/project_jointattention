from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from custom_model import CustomNet

PROJECT_PATH = Path(__file__).parents[1]
FIG_PATH = Path(PROJECT_PATH, "figures")

N_EPOCHS = 500
BATCH_SIZE = 1000
SEQ_LEN = 10
FEATURE_LEN = 5


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

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader

# model = CustomNet(
#     input_size=FEATURE_LEN,
#     seq_len=SEQ_LEN,
#     num_hiddens=128,
#     num_layers=2,
#     dropout=0.5,d
# )
# print(X.shape) for _ in range(100):
#     pred_y = model(X)


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
            
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            
            loss.backward()
            
            # perform a single optimization step (parameter update)
            optimizer.step()

            # record training loss
            train_losses.append(loss.item())


        train_loss = np.average(train_losses)
        avg_train_losses.append(train_loss)

        epoch_len = len(str(n_epochs))

        print_msg = (
            f"[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] "
            + f"train_loss: {train_loss:.5f} "
        )
        print(print_msg)

        # clear lists to track next epoch
        train_losses = []


    return model, avg_train_losses, np.array(alphas_arrs), torch.Tensor(labels).numpy()


def main():
    device = torch.device("cpu")

    model = CustomNet(
        input_size=5,
        seq_len=SEQ_LEN,
        num_hiddens=5,
        num_layers=2,
        dropout=0.1,
        attention_dim=10,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
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
    
    plt.plot(true_alphas_arr, c = 'b', label = '1, 1')
    plt.plot(false_alphas_arr, c = 'r', label = '1, 0')
    plt.legend()

    plt.savefig(Path(FIG_PATH, f'sample_attention.png'))


if __name__ == "__main__":
    main()