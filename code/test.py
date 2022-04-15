#%%
import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from custom_dataset import VideoDataset
from model import VideoRNN

SPLIT_CSV_FILE ='ija_label_train.csv'
WEIGHT_PATH = 'weights/trained.pth'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


#%%
test_transform = transforms.ToTensor()
test_dataset = VideoDataset(False, transform=test_transform)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

model = VideoRNN()
model.to(device)
model.load_state_dict(torch.load(WEIGHT_PATH))
model.eval()


#%%
def test_model(model):
    count = 0
    with torch.no_grad():
        total_acc = 0.0
        acc = 0.0
        for i, data in enumerate(test_dataloader, 0):
            inputs, label = data[0].to(device), data[1].to(device)

            outputs = model(inputs)

            acc = accuracy(outputs, label)
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