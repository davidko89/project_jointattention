#%%
import torch
import numpy as np 
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from data_loader import VideoDataset
from model import VideoRNN

PATH = './weights/trained.pth'

#%%
test_transform = transforms.ToTensor()
test_dataset = VideoDataset(False, transform=test_transform)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

model = VideoRNN()
model.to('cuda')
model.load_state_dict(torch.load(PATH))
model.eval()

#%%
def test_model(model):
    count = 0
    with torch.no_grad():
        total_acc = 0.0
        acc = 0.0
        for i, test_data in enumerate(test_dataset, 0):
            data, labels = test_data[0].to('cuda'), test_data[1].to('cuda')

            data = data.transpose(1, 3).transpose(2, 3)

            outputs = model(data)

            acc = accuracy(outputs, labels)
            total_acc += acc

            count = i

        print('avarage acc: %.5f' % (total_acc/count),'%')

    print('test finish!')


def accuracy(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


if __name__ =='__main__':
    test_model()