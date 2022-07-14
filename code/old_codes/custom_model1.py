import torch
import torch.nn as nn
from typing import Tuple


class LRCN(nn.Module):
    def __init__(self, input_size: int, seq_len: int, num_hiddens: int, num_layers: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, num_hiddens, num_layers, batch_first=True
        )  # cnn_output->input_size=512*7*7

    def forward(self, X) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        lstm_inputs = X.view(X.size(0), self.seq_len, -1)
        # print(type(lstm_inputs))
        lstm_outputs, (h, c) = self.lstm(lstm_inputs)
        return lstm_outputs, (h, c)


class CustomAttention(nn.Module):
    def __init__(self, num_hiddens):
        super().__init__()
        self.linear = nn.Linear(num_hiddens, num_hiddens)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lstm_outputs):
        # lstm_outputs (seq_len, hidden)
        alpha = self.softmax(self.tanh(self.linear(lstm_outputs)).squeeze(2))
        result = torch.sum(alpha * lstm_outputs, axis=-1)
        return result, alpha  # alpha: [batch_size, seq_len]


class CustomMLP(nn.Module):
    def __init__(self, seq_len, dropout):
        super().__init__()
        self.linear1 = nn.Linear(seq_len, seq_len)
        self.relu = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(seq_len, 2)

        self.net = nn.Sequential(
            self.linear1, self.relu, self.drop1, self.linear2,
        )

    def forward(self, X):
        return self.net(X)  # return [1, 2] batch_size, binary_classification


class CustomNet(nn.Module):
    def __init__(
        self,
        input_size,
        seq_len,
        num_hiddens,
        num_layers,
        dropout,
    ):
        super().__init__()
        self.lrcn = LRCN(
            input_size,
            seq_len,
            num_hiddens,
            num_layers,
        )
        self.attention = CustomAttention(num_hiddens)  # 128
        self.mlp = CustomMLP(seq_len, dropout)  # 150

    def forward(self, X) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        return: Y, attention
        """
        X, _ = self.lrcn(X)  # X: [1, 150, 128]
        attention, alpha = self.attention(X)
        return self.mlp(attention), alpha


if __name__ == "__main__":
    X = torch.rand(size=(1, 150, 512, 7, 7))
    # model1 = LRCN(input_size=25088, seq_len=150, num_hiddens=128, num_layers=2)
    # model2 = CustomMLP(num_hiddens=128, dropout=0.5)
    # model3 = CustomAttention(input_dimension=150)
    model4 = CustomNet(
        input_size=25088,
        seq_len=150,
        num_hiddens=128,
        num_layers=2,
        dropout=0.5,
    )

    # output = model1(X)
    # print(output[0].shape)
    # print(output[1])
    # lstm_input = X.view(X.size(0), X.size(1), -1)
    # lstm_output, hidden = model1(lstm_input)
    # print(lstm_input.shape)
    # """lstm_input=[batch_size, seq_len, input_size]"""
    # print(lstm_output.shape)
    # """hidden_state=[batch_size, seq_len, num_hiddens]"""
    # print(hidden)
    # X = model1(X)
    # pred_y = model2(X)
    # attention_embeddings = model3(X)
    Y, alpha = model4(X)
    print(Y.shape)  # Y: [1, 2]
    print(alpha.shape)  # alpha: [1, 150, 128] although it should be [1, 150]
