import torch
import torch.nn as nn
import torch.nn.functional as F
from conformer_block import *  # assumes SiLU is used inside if needed

class SpecAugment(nn.Module):
    def __init__(self, freq_mask_param=15, time_mask_param=35, num_freq_masks=2, num_time_masks=2):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def forward(self, x):
        if not self.training:
            return x  # Skip SpecAugment during evaluation

        # Handle missing channel dimension
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 1, Freq, Time)

        B, C, Freq, Time = x.shape

        for _ in range(self.num_freq_masks):
            freq_mask = torch.randint(0, self.freq_mask_param + 1, (1,)).item()
            if freq_mask < Freq:
                f0 = torch.randint(0, Freq - freq_mask + 1, (1,)).item()
                x[:, :, f0:f0+freq_mask, :] = 0

        for _ in range(self.num_time_masks):
            time_mask = torch.randint(0, self.time_mask_param + 1, (1,)).item()
            if time_mask < Time:
                t0 = torch.randint(0, Time - time_mask + 1, (1,)).item()
                x[:, :, :, t0:t0+time_mask] = 0

        return x


class ConvSubsampling(nn.Module):
    def __init__(self, in_channels=1, out_dim=144):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        # Ensure shape: (B, 1, Freq, Time)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.conv(x)  # (B, D, F', T')
        B, D, F, T = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(B, T, D * F)  # (B, T, D*F)
        return x


class TinyConformer(nn.Module):
    def __init__(self, input_dim=80, d_model=144, num_classes=8, num_blocks=4, dropout=0.1):
        super().__init__()
        self.specaug = SpecAugment()
        self.subsampling = ConvSubsampling(in_channels=1, out_dim=d_model)
        
        self.project = nn.Linear(d_model * (input_dim // 4), d_model)  # Adjusted for stride=2 twice
        self.dropout = nn.Dropout(dropout)

        self.conformers = nn.ModuleList([
            ConformerBlock(d_model=d_model, dropout=dropout) for _ in range(num_blocks)
        ])

        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (B, Freq, Time) or (B, 1, Freq, Time)
        x = self.specaug(x)  # handles (B, Freq, Time) internally
        x = self.subsampling(x)
        x = self.project(x)
        x = self.dropout(x)

        for block in self.conformers:
            x = block(x)

        x = x.transpose(1, 2)  # (B, D, T)
        x = self.pooling(x).squeeze(-1)  # (B, D)
        x = self.classifier(x)  # (B, num_classes)
        return x
