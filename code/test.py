from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from custom_dataset import create_data_loader
from model import VGG16LRCN
from pathlib import Path

# PROJECT_PATH = Path(__file__).parents[1]
# WEIGHTS_PATH = Path(PROJECT_PATH, "weights/trained.pth")

model = VGG16LRCN()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

batch_size = 4
n_epochs = 10

train_loader, test_loader, valid_loader = create_data_loader(batch_size)

# early stopping patience;
# validation loss가 개선된 마지막 시간 이후로 얼마나 기다릴지 지정
patience = 5


# test loss 및 accuracy을 모니터링하기 위해 list 초기화
test_loss = 0.0
class_correct = list(0.0 for i in range(10))
class_total = list(0.0 for i in range(10))

model.eval()  # prep model for evaluation

for data, target in test_loader:
    if len(target.data) != batch_size:
        break

    # forward pass: 입력을 모델로 전달하여 예측된 출력 계산
    output = model(data)
    # calculate the loss
    loss = criterion(output, target)
    # update test loss
    test_loss += loss.item() * data.size(0)
    # 출력된 확률을 예측된 클래스로 변환
    _, pred = torch.max(output, 1)
    # 예측과 실제 라벨과 비교
    correct = np.squeeze(pred.eq(target.data.view_as(pred)))
    # 각 object class에 대해 test accuracy 계산
    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# calculate and print avg test loss
test_loss = test_loss / len(test_loader.dataset)
print("Test Loss: {:.6f}\n".format(test_loss))

for i in range(10):
    if class_total[i] > 0:
        print(
            "Test Accuracy of %5s: %2d%% (%2d/%2d)"
            % (
                str(i),
                100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]),
                np.sum(class_total[i]),
            )
        )
    else:
        print("Test Accuracy of %5s: N/A (no training examples)" % (class_total[i]))

print(
    "\nTest Accuracy (Overall): %2d%% (%2d/%2d)"
    % (
        100.0 * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct),
        np.sum(class_total),
    )
)
