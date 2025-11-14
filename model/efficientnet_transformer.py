import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class EfficientNetBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.net = EfficientNet.from_pretrained('efficientnet-b0') if pretrained else EfficientNet.from_name('efficientnet-b0')
        self.out_dim = self.net._fc.in_features

        self.net._fc = nn.Identity()

    def forward(self, x):
        B, C, T, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(B * T, C, H, W)
        x = self.net(x)
        x = x.reshape(B, T, -1)
        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, in_dim=1280, embed_dim=512, num_positions=16):
        super().__init__()
        self.proj = nn.Linear(in_dim, embed_dim)
        self.scale = embed_dim ** -0.5    # ổn định hơn

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_positions + 1, embed_dim))

        # self.norm_pre = nn.LayerNorm(embed_dim)
        self.pos_drop = nn.Dropout(0.1)

    def forward(self, x):
        B, T, _ = x.size()

        x = self.proj(x) * self.scale          # (B, T, embed_dim)

        cls = self.cls_token.expand(B, -1, -1) # (B, 1, embed_dim)
        x = torch.cat((cls, x), dim=1)

        x = x + self.pos_embed[:, :T + 1, :]
        # x = self.norm_pre(x)
        x = self.pos_drop(x)
        return x
    

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=512, num_heads=4, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x):
        x_norm = self.norm1(x)
        x_attn, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + x_attn

        x_mlp = self.mlp(self.norm2(x))
        x = x + x_mlp

        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=512, depth=2, num_heads=4):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])

    def forward(self, x):
        for blk in self.layers:
            x = blk(x)
        return x
    
class efficientnet_transformer(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, num_frames=16, embed_dim=512, depth=2, num_heads=4):
        super().__init__()

        self.backbone = EfficientNetBackbone(pretrained=pretrained)
        im_dim = self.backbone.out_dim

        self.embedding = PositionalEmbedding(im_dim, embed_dim, num_frames)
        self.transformer = TransformerEncoder(embed_dim, depth, num_heads)

        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

        self.apply(self._init_weights)

        if not pretrained:
            self._init_backbone(self.backbone)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

        elif isinstance(m, nn.MultiheadAttention):
            nn.init.xavier_uniform_(m.in_proj_weight)
            if m.in_proj_bias is not None:
                nn.init.zeros_(m.in_proj_bias)

            nn.init.xavier_uniform_(m.out_proj.weight)
            if m.out_proj.bias is not None:
                nn.init.zeros_(m.out_proj.bias)

    def _init_backbone(self, module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.backbone(x)
        x = self.embedding(x)
        x = self.transformer(x)

        cls = x[:, 0]
        cls = self.norm(cls)
        logits = self.fc(cls)
        return logits

