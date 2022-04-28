import torch
import torch.nn as nn
import torch.nn.functional as F



DROPOUT = 0.4
SEQ_LEN = 300
NUM_LSTM_LAYERS = 1
LSTM_HIDDEN_DIM = 128


class LRCN(nn.Module):
    def __init__(self, model_name):
        super(LRCN, self).__init__()
        self.lstm_hidden_dim = LSTM_HIDDEN_DIM
        self.num_lstm_layers = NUM_LSTM_LAYERS
        self.seq_len = SEQ_LEN
        self.model_name = model_name
        self.drop1 = nn.Dropout(DROPOUT)
        self.lstm1 = nn.LSTM(
            25088, self.lstm_hidden_dim, self.num_lstm_layers, batch_first=True
        )  # input features = 512 * 7 * 7
        self.linear1 = nn.Linear(self.lstm_hidden_dim, 512)
        self.linear2 = nn.Linear(512, 2)

    def forward(self, X):
        X = X.view(X.size(0), -1)
        # 1st dim of tensor(batch_size) to vector
        # make output as (batch_size, seq_len, output_size)
        X = X.view(X.size(0), self.seq_len, -1)
        out, (h_n, c_n) = self.lstm1(X)
        out = out[:, -1]  # get output of the last lstm
        out = self.linear1(out)
        out = F.relu(out)
        out = self.drop1(out)
        out = self.linear2(out)

        return out


if __name__ == "__main__":
    # X = torch.rand(size=(1, 300, 512, 7, 7))
    # print(X)
    # print(X.shape)
    # print(X.view(-1).shape)

    model = LRCN(model_name="vgg16lrcn")
    model.to("cuda")
    X = torch.rand(size=(100, 300, 512, 7, 7))
    X = X.to("cuda")
    pred_y = model(X)
