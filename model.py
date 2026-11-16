
import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()

        self.padding = 2 * dilation

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            dilation=dilation
        )

    def forward(self, x):
        x = torch.nn.functional.pad(
            x, (self.padding, 0)
        )
        return self.conv(x)
class SimpleTCN(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            CausalConv1d(51, 32, dilation=1),
            nn.ReLU(),

            CausalConv1d(32, 32, dilation=2),
            nn.ReLU(),

            CausalConv1d(32, 32, dilation=4),
            nn.ReLU(),

            CausalConv1d(32, 32, dilation=8),
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
