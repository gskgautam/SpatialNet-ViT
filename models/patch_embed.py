import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """
    Patch Embedding module for Vision Transformer (ViT).
    Splits image into patches and projects to embedding dimension.
    """
    def __init__(self, img_size=256, patch_size=16, in_chans=3, embed_dim=512):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x 