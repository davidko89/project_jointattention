import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


DROPOUT = 0.4
BATCH_SIZE = 2
SEQ_LEN = 300
NUM_LSTM_LAYERS = 1
LSTM_HIDDEN_DIM = 128


class CNNLRCN(nn.Module):
    def __init__(self, model_name):
        super(CNNLRCN, self).__init__()
        self.lstm_hidden_dim = LSTM_HIDDEN_DIM
        self.num_lstm_layers = NUM_LSTM_LAYERS
        self.batch_size = BATCH_SIZE
        self.seq_len = SEQ_LEN
        self.model_name = model_name

        # if self.model_name == 'vgg':
        self.basemodel = models.vgg16(pretrained=True).features
        # self.basemodel = models.vgg16(pretrained=True).features

        # print("Loaded pretrained VGG16 weights")

        for param in self.basemodel.parameters():
            param.requires_grad = False

        # number of features = 512 * 7 * 7

        self.drop1 = nn.Dropout(DROPOUT)

        self.lstm1 = nn.LSTM(
            25088, self.lstm_hidden_dim, self.num_lstm_layers, batch_first=True
        )
        self.linear1 = nn.Linear(self.lstm_hidden_dim, 512)
        self.linear2 = nn.Linear(512, 2)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        # reshape input to be (batch_size * seq_len, input_size)
        x = x.view(batch_size * seq_len, c, h, w)

        out = self.basemodel(x)

        out = out.view(
            out.size(0), -1
        )  # 1st dimension of tensor (batch_size) to vector

        # make output as (batch_size, seq_len, output_size)
        out = out.view(batch_size, seq_len, -1)

        out, (h_n, c_n) = self.lstm1(out)

        out = out[:, -1]  # get output of the last lstm

        out = self.linear1(out)
        out = F.relu(out)

        out = self.drop1(out)
        out = self.linear2(out)

        return out


if __name__ == "__main__":
    # vgg = models.vgg16(pretrained=True)
    # output = vgg.features(torch.rand(size=(300, 3, 224, 224)))
    # print(vgg.features)
    # print(output.shape)
    # print(output.view(-1).shape)
    model = CNNLRCN()
    arr = torch.rand(size=(BATCH_SIZE, 300, 3, 224, 224))
    # arr = arr.to("cuda")
    # model.to("cuda")
    # pred_y = model(arr)
