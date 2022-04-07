#%%
import matplotlib.pyplot as plt
import numpy as np 
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from data_loader import train_dataset
from model import VideoRNN
import torch.optim as optim

PATH = 'weights/trained.pth'
#%%
x_train = np.load('./dataset/x_train.npy').astype(np.float32)  # (2586, 26, 34, 1)
y_train = np.load('./dataset/y_train.npy').astype(np.float32)  # (2586, 1)

train_transform = transforms.Compose([transforms.ToTensor()])
train_dataset = train_dataset(x_train, y_train, transform=train_transform)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

model = VideoRNN()
model.to('cuda')
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
epochs = 50

#%%
def train_model(model, criterion, optimizer): 
    for epoch in range(epochs):
        running_loss = 0.0
        running_acc = 0.0

        for i, data in enumerate(train_dataset, 0):
            input_1, labels = data[0].to('cuda'), data[1].to('cuda')

            input = input_1.transpose(1, 3).transpose(2, 3)

            optimizer.zero_grad()

            outputs = model(input)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_acc += accuracy(outputs, labels)

            if i % 80 == 79:
                print('epoch: [%d/%d] train_loss: %.5f train_acc: %.5f' % (
                    epoch + 1, epochs, running_loss / 80, running_acc / 80))
                running_loss = 0.0

    print("learning finish")
    torch.save(model.state_dict(), PATH)


def accuracy(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


if __name__ =='__main__':
    train_model()