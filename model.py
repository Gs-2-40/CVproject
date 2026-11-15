
import torch
import torch.nn as nn


class SimpleTCN(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv1d(51, 32, 3, dilation=1),
            nn.ReLU(),

            nn.Conv1d(32, 32, 3, dilation=2),
            nn.ReLU(),

            nn.Conv1d(32, 32, 3, dilation=4),
            nn.ReLU(),

            nn.Conv1d(32, 32, 3, dilation=8),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(32, 2)

    def forward(self, x):
        # (B, 30, 17, 3) -> (B, 51, 30)
        b, t, j, f = x.shape
        x = x.reshape(b, t, j * f).transpose(1, 2)

        x = self.network(x)

        # Последний временной шаг
        x = x[:, :, -1]

        return self.classifier(x)
