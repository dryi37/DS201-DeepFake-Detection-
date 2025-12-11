import torch
import torch.nn as nn
from torchvision import models


def extract_grid_patches(x: torch.Tensor, grid_size: int = 4):
    B, C, H, W = x.shape
    assert H % grid_size == 0 and W % grid_size == 0, \
        f"H,W ({H},{W}) phải chia hết cho grid_size={grid_size}"

    patch_h = H // grid_size
    patch_w = W // grid_size

    # unfold: [B, C, grid_h, patch_h, grid_w, patch_w]
    patches = x.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)

    # [B, grid_h, grid_w, C, h, w]
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    B, gh, gw, C, h, w = patches.shape

    # [B, N, C, h, w]
    patches = patches.view(B, gh * gw, C, h, w)
    return patches


class LocalPatchEncoder(nn.Module):
    def __init__(
        self,
        grid_size: int = 4,
        backbone_name: str = "efficientnet_b1",   # chỉ dùng EfficientNet
        backbone_out_dim: int = 512,
        pretrained_backbone: bool = False,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.num_patches = grid_size * grid_size
        self.backbone_out_dim = backbone_out_dim
        self.backbone_name = backbone_name

        if backbone_name == "efficientnet_b0":
            backbone = models.efficientnet_b0(pretrained=pretrained_backbone)
        elif backbone_name == "efficientnet_b1":
            backbone = models.efficientnet_b1(pretrained=pretrained_backbone)
        elif backbone_name == "efficientnet_b2":
            backbone = models.efficientnet_b2(pretrained=pretrained_backbone)
        else:
            raise ValueError(f"Unsupported EfficientNet backbone: {backbone_name}")

        in_dim = backbone.classifier[1].in_features
        backbone.classifier[1] = nn.Linear(in_dim, backbone_out_dim)
        self.backbone = backbone

    def forward(self, x: torch.Tensor):
        patches = extract_grid_patches(x, grid_size=self.grid_size)
        B, N, C, h, w = patches.shape

        patches = patches.view(B * N, C, h, w)        # [B*N, C, h, w]
        patch_feats = self.backbone(patches)          # [B*N, D]
        patch_feats = patch_feats.view(B, N, -1)      # [B, N, D]

        return patch_feats


class GlobalEncoder(nn.Module):
    def __init__(
        self,
        backbone_name: str = "efficientnet_b1",
        backbone_out_dim: int = 512,
        pretrained_backbone: bool = False,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone_out_dim = backbone_out_dim

        if backbone_name == "efficientnet_b0":
            backbone = models.efficientnet_b0(pretrained=pretrained_backbone)
        elif backbone_name == "efficientnet_b1":
            backbone = models.efficientnet_b1(pretrained=pretrained_backbone)
        elif backbone_name == "efficientnet_b2":
            backbone = models.efficientnet_b2(pretrained=pretrained_backbone)
        else:
            raise ValueError(f"Unsupported EfficientNet backbone: {backbone_name}")

        in_dim = backbone.classifier[1].in_features
        backbone.classifier[1] = nn.Linear(in_dim, backbone_out_dim)
        self.backbone = backbone

    def forward(self, x: torch.Tensor):
        global_feat = self.backbone(x)
        return global_feat


class LocalGlobalFusionTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_patches: int = 16,   # grid_size=4 -> 16 patch
    ):
        super().__init__()
        self.d_model = d_model
        self.max_patches = max_patches

        # 1 global token + max_patches local token
        max_tokens = 1 + max_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,   # [B, T, D]
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, global_feat: torch.Tensor, patch_feats: torch.Tensor):
        B, D = global_feat.shape
        B2, N, D2 = patch_feats.shape
        assert B == B2 and D == D2, "global_feat và patch_feats phải cùng batch & dim"
        assert N <= self.max_patches, f"Số patch ({N}) > max_patches ({self.max_patches})"

        # ghép global token + local tokens
        global_tok = global_feat.unsqueeze(1)          # [B, 1, D]
        tokens = torch.cat([global_tok, patch_feats], dim=1)  # [B, 1+N, D]

        T = tokens.size(1)
        pos = self.pos_embed[:, :T, :]                 # [1, T, D]
        tokens = tokens + pos

        fused_tokens = self.encoder(tokens)            # [B, T, D]
        fused_frame_feat = fused_tokens[:, 0, :]       # dùng global token sau fusion

        return fused_frame_feat


class TemporalBiLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,        # D
        hidden_size: int = 512,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,        # [B, T, D]
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_out_dim = hidden_size * (2 if bidirectional else 1)
        self.lstm_out_dim = lstm_out_dim
        self.out_dim = lstm_out_dim

    def forward(self, frame_feats: torch.Tensor):
        out, (h_n, c_n) = self.lstm(frame_feats)
        # mean pooling theo thời gian
        video_feat = out.mean(dim=1)  # [B, lstm_out_dim]
        return video_feat
    
class DeepfakeDetectionModel(nn.Module):
    def __init__(
        self,
        grid_size: int = 4,
        backbone_name: str = "efficientnet_b1",
        backbone_out_dim: int = 512,
        pretrained_backbone: bool = False,
        fusion_layers: int = 1,
        fusion_heads: int = 8,
        fusion_dropout: float = 0.1,
        temporal_hidden_size: int = 256,
        temporal_layers: int = 1,
        temporal_bidirectional: bool = True,
        temporal_dropout: float = 0.1,
    ):
        super().__init__()

        self.local_encoder = LocalPatchEncoder(
            grid_size=grid_size,
            backbone_name=backbone_name,
            backbone_out_dim=backbone_out_dim,
            pretrained_backbone=pretrained_backbone,
        )

        self.global_encoder = GlobalEncoder(
            backbone_name=backbone_name,
            backbone_out_dim=backbone_out_dim,
            pretrained_backbone=pretrained_backbone,
        )

        self.local_proj = nn.Sequential(
            nn.Linear(backbone_out_dim, backbone_out_dim),
            nn.LayerNorm(backbone_out_dim),
        )
        self.global_proj = nn.Sequential(
            nn.Linear(backbone_out_dim, backbone_out_dim),
            nn.LayerNorm(backbone_out_dim),
        )

        self.fusion = LocalGlobalFusionTransformer(
            d_model=backbone_out_dim,
            num_layers=fusion_layers,
            num_heads=fusion_heads,
            dropout=fusion_dropout,
            max_patches=grid_size * grid_size,
        )

        self.temporal = TemporalBiLSTM(
            input_dim=backbone_out_dim,
            hidden_size=temporal_hidden_size,
            num_layers=temporal_layers,
            bidirectional=temporal_bidirectional,
            dropout=temporal_dropout,
        )

        # Classifier head (BCEWithLogitsLoss dùng logits, không sigmoid ở đây)
        self.classifier = nn.Sequential(
            nn.Linear(self.temporal.out_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
        )

    def forward(self, x: torch.Tensor):
        B, C, T, H, W = x.shape

        x_btchw = x.permute(0, 2, 1, 3, 4).contiguous()     # [B, T, C, H, W]
        x_flat = x_btchw.view(B * T, C, H, W)               # [B*T, C, H, W]

        # local + global
        patch_feats = self.local_encoder(x_flat)            # [B*T, N, D]
        global_feats = self.global_encoder(x_flat)          # [B*T, D]

        patch_feats = self.local_proj(patch_feats)          # [B*T, N, D]
        global_feats = self.global_proj(global_feats)       # [B*T, D]

        # fusion
        fused_flat = self.fusion(global_feats, patch_feats) # [B*T, D]

        # reshape lại thành sequence theo thời gian
        D = fused_flat.size(-1)
        frame_feats = fused_flat.view(B, T, D)              # [B, T, D]

        # temporal
        video_feat = self.temporal(frame_feats)             # [B, D_video]

        # logits
        logits = self.classifier(video_feat)    # [B, 2]

        return logits
