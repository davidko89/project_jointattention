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
        lstm_outputs, hidden_states = self.lstm(lstm_inputs)
        return lstm_outputs, hidden_states


class CustomAttention(nn.Module):
    def __init__(self, num_hiddens, attention_dim):
        super().__init__()
        self.lstmoutput_attention_projection = nn.Linear(num_hiddens, attention_dim)
        self.attention = nn.Linear(attention_dim, 1)
        self.ReLU = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lstm_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """compute z_t but will use only alphas_t for visualization @ testing
        Args:
        lstm_outputs (torch.Tensor): [batch_size, seq_len, num_hiddens]
        """
        lstmoutput_attention = self.lstmoutput_attention_projection(lstm_outputs)
        # In: (batch_dim, seq_len, num_hiddens), Out: (batch_dim, seq_len, attention_dim)
        attention = self.attention(self.ReLU(lstmoutput_attention)).squeeze(2)
        # In: (batch_dim, seq_len, attention_dim), Out: (batch_size, seq_len)
        alphas_t = self.softmax(attention)  # Out: (batch_dim, seq_len)
        attention_weighted_encoding = (lstm_outputs * alphas_t.unsqueeze(2)).sum(
            dim=1
        )  # Out: (batch_diim, num_hiddens)
        return attention_weighted_encoding, alphas_t


class CustomMLP(nn.Module):
    def __init__(self, num_hiddens, dropout):
        super().__init__()
        self.linear1 = nn.Linear(num_hiddens, num_hiddens)
        self.relu = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(num_hiddens, 2)
        # self.softmax = nn.Softmax(dim=-1)

        self.net = nn.Sequential(
            self.linear1, self.relu, self.drop1, self.linear2
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
        attention_dim,
    ):
        super().__init__()
        self.lrcn = LRCN(
            input_size,
            seq_len,
            num_hiddens,
            num_layers,
        )
        self.attention = CustomAttention(num_hiddens, attention_dim)
        self.mlp = CustomMLP(num_hiddens, dropout)

    def forward(self, X) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input: X (=lstm_outputs: torch.Tensor)
        Returns: output, attention
        """
        X, _ = self.lrcn(X)  # X: [1, 150, 128]
        z_t, alphas_t = self.attention(X)
        return self.mlp(z_t), alphas_t


if __name__ == "__main__":
    X = torch.rand(size=(1, 150, 512, 7, 7))
    # model1 = LRCN(input_size=25088, seq_len=150, num_hiddens=128, num_layers=1)
    # lstm_outputs, hidden_states = model1(X)
    # print(X.shape)
    # print(lstm_outputs.shape)
    # print(hidden_states[0].shape)

    # model2 = CustomAttention(num_hiddens=128, attention_dim=128)
    # z_t, alphas_t = model2(lstm_outputs)
    # print(z_t.shape)
    # print(alphas_t.shape)

    # model3 = CustomMLP(num_hiddens=128, dropout=0.5)

    model4 = CustomNet(
        input_size=25088,
        seq_len=150,
        num_hiddens=128,
        num_layers=2,
        dropout=0.4,
        attention_dim=128,
    )
    out, alphas_t = model4(X)
    print(out.shape)  # out: [1, 2]
    print(alphas_t.shape)  # alphas_t: [1, 150]