import torch
import torch.nn as nn
from models.vit_backbone import ViTBackbone
from models.multitask_heads import LandUseClassificationHead, CaptionGenerationHead, VQAClassificationHead, CountingHead

class SpatialNetViT(nn.Module):
    """
    Full SpatialNet-ViT model: ViT backbone + multi-task heads.
    Routes to the correct head based on the 'task' argument.
    task: one of ['landuse', 'caption', 'vqa_presence', 'vqa_comparison', 'vqa_urbanrural', 'count']
    """
    def __init__(self, embed_dim=512, num_classes=21, vocab_size=30522):
        super().__init__()
        self.vit = ViTBackbone(embed_dim=embed_dim)
        self.landuse_head = LandUseClassificationHead(embed_dim, num_classes)
        self.caption_head = CaptionGenerationHead(embed_dim, vocab_size)
        self.vqa_presence_head = VQAClassificationHead(embed_dim, 2)
        self.vqa_comparison_head = VQAClassificationHead(embed_dim, 2)
        self.vqa_urbanrural_head = VQAClassificationHead(embed_dim, 2)
        self.count_head = CountingHead(embed_dim)
    def forward(self, x, task, tgt=None, tgt_mask=None):
        vit_out = self.vit(x)  # (B, N+1, D)
        cls_token = vit_out[:, 0]  # (B, D)
        if task == 'landuse':
            return self.landuse_head(cls_token)
        elif task == 'caption':
            # tgt and tgt_mask required for captioning
            return self.caption_head(vit_out, tgt, tgt_mask)
        elif task == 'vqa_presence':
            return self.vqa_presence_head(cls_token)
        elif task == 'vqa_comparison':
            return self.vqa_comparison_head(cls_token)
        elif task == 'vqa_urbanrural':
            return self.vqa_urbanrural_head(cls_token)
        elif task == 'count':
            return self.count_head(cls_token)
        else:
            raise ValueError(f'Unknown task: {task}') 