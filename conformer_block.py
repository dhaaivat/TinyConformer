import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForwardModule(nn.Module):
    def __init__(self, d_model, expansion_factor=4, dropout=0.1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * expansion_factor),
            nn.Swish(),  # or GELU
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion_factor, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + 0.5 * self.layer(x)

class ConvolutionModule(nn.Module):
    def __init__(self, d_model, kernel_size=31, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size,
                                        groups=d_model, padding=kernel_size // 2)
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Input shape: (B, T, D)
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # (B, D, T)
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = F.silu(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)  # (B, T, D)
        return x

class ConformerBlock(nn.Module):
    def __init__(self, d_model=144, n_heads=4, ffn_expansion=4, conv_kernel_size=31, dropout=0.1):
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model, ffn_expansion, dropout)
        
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.self_attn_dropout = nn.Dropout(dropout)

        self.conv_module = ConvolutionModule(d_model, kernel_size=conv_kernel_size, dropout=dropout)

        self.ffn2 = FeedForwardModule(d_model, ffn_expansion, dropout)

        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None):
        # x: (B, T, D)
        x = self.ffn1(x)

        # MHSA
        residual = x
        x = self.self_attn_layer_norm(x)
        attn_output, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = residual + self.self_attn_dropout(attn_output)

        # Conv Module
        x = x + self.conv_module(x)

        # FFN2
        x = self.ffn2(x)

        # Final Norm
        x = self.final_layer_norm(x)
        return x