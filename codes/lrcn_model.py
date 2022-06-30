import torch
import torch.nn as nn
import torch.nn.functional as F


class LRCN(nn.Module):
    def __init__(self, model_name, dropout, seq_len, lstm_layers, hidden_size):
        super(LRCN, self).__init__()
        self.model_name = model_name
        self.seq_len = seq_len
        self.lstm = nn.LSTM(
            25088, hidden_size, lstm_layers, batch_first=True
        )  # input features = 512 * 7 * 7 / batch_size placed @ 1st dim
        self.linear1 = nn.Linear(hidden_size, 512)
        self.drop1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(512, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, conv_output):
        lstm_input = conv_output.view(conv_output.size(0), self.seq_len, -1)
        # get lstm_input as (batch_size, seq_len, output_size)
        lstm_output, (h_n, c_n) = self.lstm(lstm_input)
        lstm_output = lstm_output[:, -1, :]  # get output of the last lstm
        output1 = self.linear1(lstm_output)
        output_relu = F.relu(output1)
        output_drop = self.drop1(output_relu)
        output2 = self.linear2(output_drop)
        output = self.softmax(output2)
        return output  

# if __name__ == "__main__":
    # X = torch.rand(size=(1, 300, 512, 7, 7))
    # print(X)
    # print(X.shape)
    # print(X.view(-1).shape)
    
    # model = LRCN()
    # model.to("cuda")
    # conv_output = torch.rand(size=(100, 300, 512, 7, 7))
    # conv_output = conv_output.to("cuda")
    # pred_y = model(conv_output)
