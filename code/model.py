#%%
import torch.nn as nn
import torchvision.models as models

class CNN(nn.Module):
    def __init__(self, train_CNN=False, num_classes=1):
        super(CNN, self).__init__()
        self.train_CNN = train_CNN
        self.vgg16 = models.vgg16(pretrained=True, aux_logits=False)
        self.vgg16.fc = nn.Linear(self.inception.fc.in_features, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax()

    def forward(self, images):
        features = self.vgg16(images)
        return self.softmax(self.dropout(self.relu(features))).squeeze(1)