import torch
from torch import nn
from torch.autograd import Variable

device = "cuda" if torch.cuda.is_available() else "cpu"


class BidLstm(nn.Module):
    def __init__(self, input_dim, output_size, config):
        super().__init__()
        self.input = input_dim
        self.output = output_size
        self.config = config
        self.hidden_dim = config.hidden_size

        self.fc = self.make_fc()

        self.lstm = nn.LSTM(
            self.input,
            self.hidden_dim // 2,
            self.config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.config.dropout,
        )

    def make_fc(self):
        layers = [
            nn.BatchNorm1d(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
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
            torch.zeros(
                self.config.num_layers * 2,
                x.size(0),
                self.hidden_dim // 2,
                device=device,
            )
        )
        c0 = Variable(
            torch.zeros(
                self.config.num_layers * 2,
                x.size(0),
                self.hidden_dim // 2,
                device=device,
            )
        )
        out, (h_out, _) = self.lstm(x, (h0, c0))

        y_pred = self.fc(out[:, -1, :])

        return y_pred
