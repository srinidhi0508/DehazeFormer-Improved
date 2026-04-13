import torch
import torch.nn as nn
import torch.nn.functional as F

# -------- EDGE --------
def get_edge(x):
    B, C, H, W = x.shape

    sobel_x = torch.tensor(
        [[[-1, 0, 1],
          [-2, 0, 2],
          [-1, 0, 1]]],
        dtype=torch.float32
    ).to(x.device)

    sobel_y = torch.tensor(
        [[[-1, -2, -1],
          [ 0,  0,  0],
          [ 1,  2,  1]]],
        dtype=torch.float32
    ).to(x.device)

    # Expand filters to match channels
    sobel_x = sobel_x.expand(C, 1, 3, 3)
    sobel_y = sobel_y.expand(C, 1, 3, 3)

    # Apply grouped convolution (VERY IMPORTANT)
    edge_x = F.conv2d(x, sobel_x, padding=1, groups=C)
    edge_y = F.conv2d(x, sobel_y, padding=1, groups=C)

    edge = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)

    # Reduce to 1 channel
    edge = torch.mean(edge, dim=1, keepdim=True)

    return edge

# -------- ILLUMINATION --------
def get_illumination(x):
    return torch.mean(x, dim=1, keepdim=True)
     

# -------- MULTI-CUE ATTENTION --------
class MultiCueAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(channels + 2, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        edge = get_edge(x)       # (B,1,H,W)
        illum = get_illumination(x)  # (B,1,H,W)

        concat = torch.cat([x, edge, illum], dim=1)

        weight = self.conv(concat)
        return x * weight