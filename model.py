import torch.nn as nn


class NextProfile1D(nn.Module):
    def __init__(self, n_hist=5):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=n_hist, out_channels=32, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=7, padding=3)
        self.conv3 = nn.Conv1d(32, 16, kernel_size=5, padding=2)
        self.out = nn.Conv1d(16, 1,  kernel_size=3, padding=1)

        self.dropout = nn.Dropout(0.15)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=32)
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=32)
        self.gn3 = nn.GroupNorm(num_groups=8, num_channels=16)

        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [batch_size, n_hist, 100]
        x = self.relu(self.gn1(self.conv1(x)))
        x = self.relu(self.gn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.relu(self.gn3(self.conv3(x)))
        x = self.out(x)  # [batch_size, 1, 100]
        return x.squeeze(1)  # [batch_size, 100]