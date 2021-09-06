import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"

"""
Torch Simple Lstm
auth: Methodfunc - Kwak Piljong
date: 2021.08.26
version: 0.1
"""


class SimpleLstm(nn.Module):
    def __init__(self, input_dim, hidden_size, output_size, num_layers, batch_size):
        super().__init__()
        self.input = input_dim
        self.hidden_dim = hidden_size
        self.output = output_size
        self.layers = num_layers
        self.batch_size = batch_size

        self.hidden_layer = self.init_hidden()
        self.fc = self.make_fc()

        self.lstm = nn.LSTM(self.input, self.hidden_dim, self.layers, batch_first=True)

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
        lstm_out, self.hidden_layer = self.lstm(x, self.hidden_layer)
        y_pred = self.fc(lstm_out[-1].view(self.batch_size, -1))

        return y_pred
