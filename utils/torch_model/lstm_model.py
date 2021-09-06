import torch
from torch import nn
from torch.autograd import Variable

device = "cuda" if torch.cuda.is_available() else "cpu"

"""
LSTM model
auth: Methodfunc - Kwak Piljong
date: 2021.08.27
version: 0.1
"""


class LstmModel(nn.Module):
    def __init__(self, input_dim, hidden_size, output_size, num_layers, batch_size):
        super().__init__()
        self.input = input_dim
        self.hidden_dim = hidden_size
        self.output = output_size
        self.layers = num_layers
        self.batch_size = batch_size

        self.hidden_layer = self.init_hidden()
        self.fc = self.make_fc()

        self.lstm = nn.LSTM(
            self.input,
            self.hidden_dim,
            self.layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        self.lstm_2 = nn.LSTM(
            self.hidden_dim * 2,
            self.hidden_dim // 2,
            1,
            bidirectional=True,
            dropout=0.2,
        )

    def init_hidden(self):
        return (
            torch.zeros(self.layers, self.batch_size, self.hidden_dim, device=device),
            torch.zeros(self.layers, self.batch_size, self.hidden_dim, device=device),
        )

    def make_fc(self):
        layers = [
            nn.BatchNorm1d(self.hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 4, self.output),
        ]

        reg = nn.Sequential(*layers)

        return reg

    def forward(self, x):
        h0 = Variable(
            torch.zeros(self.layers * 2, x.size[0], self.hidden_dim, device=device)
        )
        c0 = Variable(
            torch.zeros(self.layers * 2, x.size[0], self.hidden_dim, device=device)
        )
        lstm_out, _ = self.lstm(x, (h0, c0))

        h1 = Variable(
            torch.zeros(self.layers, x.size[0], self.hidden_dim // 2, device=device)
        )
        c1 = Variable(
            torch.zeros(self.layers, x.size[0], self.hidden_dim // 2, device=device)
        )

        lstm_out_2, _ = self.lstm_2(lstm_out, (h1, c1))
        y_pred = self.fc(lstm_out_2[-1].view(self.batch_size, -1))

        return y_pred
