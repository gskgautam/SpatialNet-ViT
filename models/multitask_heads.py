import torch
import torch.nn as nn

class LandUseClassificationHead(nn.Module):
    """
    Head for land-use classification (FC + softmax)
    """
    def __init__(self, embed_dim=512, num_classes=21):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)
    def forward(self, x):
        # x: (B, D) (CLS token)
        return self.fc(x)

class CaptionGenerationHead(nn.Module):
    """
    Head for caption generation (Transformer decoder + FC)
    """
    def __init__(self, embed_dim=512, vocab_size=30522, num_layers=2, num_heads=8, max_len=32):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
    def forward(self, memory, tgt, tgt_mask=None):
        # memory: (B, N, D) from ViT backbone
        # tgt: (B, T) token ids
        tgt_emb = self.token_emb(tgt)
        out = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        return self.fc_out(out)

class VQAClassificationHead(nn.Module):
    """
    Head for VQA classification (FC + softmax)
    """
    def __init__(self, embed_dim=512, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)
    def forward(self, x):
        # x: (B, D) (CLS token)
        return self.fc(x)

class CountingHead(nn.Module):
    """
    Head for counting (FC + linear)
    """
    def __init__(self, embed_dim=512):
        super().__init__()
        self.fc = nn.Linear(embed_dim, 1)
    def forward(self, x):
        # x: (B, D) (CLS token)
        return self.fc(x) 