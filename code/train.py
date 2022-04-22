from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from custom_dataset_diagnosis import VideoDataset
import model
from model import TimeWarp, extractlastcell
import torchvision.models as models

SPLIT_CSV_FILE = "ija_diagnosis_train.csv"
WEIGHT_PATH = "weights/trained.pth"
NUM_EPOCHS = 10
BATCH_SIZE = 2
LEARNING_RATE = 0.001

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


train_transform = transforms.ToTensor()
train_dataset = VideoDataset(True, SPLIT_CSV_FILE, train_transform)
train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
)


def train_model(model, criterion, optimizer):
    for epoch in tqdm(range(NUM_EPOCHS)):
        running_loss = 0.0
        running_acc = 0.0

        for batch_idx, (X, y) in enumerate(train_dataloader):
            # get the inputs; data is a list of [x, y]
            X = X.to(device)
            # print(X.shape)
            y = y.int().to(device)

            print(y.dtype)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            running_acc += accuracy(outputs, y)

            if batch_idx % BATCH_SIZE == BATCH_SIZE - 1:
                print(
                    "epoch: [%d/%d] train_loss: %.5f train_acc: %.5f"
                    % (
                        epoch + 1,
                        NUM_EPOCHS,
                        running_loss / len(train_dataloader),
                        running_acc / len(train_dataloader),
                    )
                )

    print("learning finish")
    torch.save(model.state_dict(), WEIGHT_PATH)


def accuracy(y_pred, y_test):
    y_pred_tag = torch.sigmoid(y_pred)

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


if __name__ == "__main__":
    train_model(model, criterion, optimizer)
