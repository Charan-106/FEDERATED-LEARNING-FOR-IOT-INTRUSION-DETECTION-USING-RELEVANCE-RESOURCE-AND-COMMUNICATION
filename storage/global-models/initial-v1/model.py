import torch
import torch.nn as nn


class PreActResBlock(nn.Module):
    def __init__(self, width, dropout=0.2):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(width)
        self.fc1 = nn.Linear(width, width)
        self.bn2 = nn.BatchNorm1d(width)
        self.fc2 = nn.Linear(width, width)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(p=dropout)

        nn.init.zeros_(self.bn2.weight)

    def forward(self, x):
        out = self.fc1(self.act(self.bn1(x)))
        out = self.fc2(self.dropout(self.act(self.bn2(out))))
        return x + out


class ResMLP(nn.Module):
    def __init__(self, in_dim, n_classes, hidden=None, dropout=0.2):
        super().__init__()
        hidden = hidden or [768, 768, 768, 768]
        width = hidden[0]

        self.in_bn = nn.BatchNorm1d(in_dim)

        self.stem = nn.Sequential(
            nn.Linear(in_dim, width),
            nn.BatchNorm1d(width),
            nn.SiLU(),
            nn.Dropout(p=dropout)
        )

        self.blocks = nn.Sequential(
            *[PreActResBlock(width, dropout) for _ in hidden]
        )

        self.out_bn = nn.BatchNorm1d(width)

        self.head = nn.Sequential(
            nn.SiLU(),
            nn.Linear(width, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        x = self.stem(self.in_bn(x))
        x = self.blocks(x)
        return self.head(self.out_bn(x))