import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet


# =========================
# 0. EfficientNet Backbone (CNN-base)
# =========================
class EfficientNetBackbone(nn.Module):
    """
    Dùng EfficientNet làm backbone spatial, lấy feature map 2D (không GAP).
    """
    def __init__(self, model_name: str = "efficientnet-b0", pretrained: bool = True, in_channels: int = 3):
        super().__init__()

        self.net = EfficientNet.from_pretrained(model_name) if pretrained else EfficientNet.from_name(model_name)
        # Nếu input không phải 3 channel (ví dụ thêm optical flow), sửa conv đầu
        if in_channels != 3:
            old_conv = self.net._conv_stem
            self.net._conv_stem = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
        # Số channel cuối cùng (B0: 1280)
        self.out_channels = self.net._fc.in_features

    def forward(self, x):
        # x: (B*, C, H, W)
        # extract_features trả ra feature map (B*, C_out, H', W')
        return self.net.extract_features(x)


# =========================
# 1. Artifact Amplification Block (AAB)
# =========================
class ArtifactAmplificationBlock(nn.Module):
    """
    AAB: học 1 mask (0-1) để khuếch đại vùng nghi là artifact.
    Input:  (B*, C, H, W)
    Output: (B*, C, H, W)
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        mask = self.conv1(x)
        mask = self.bn1(mask)
        mask = F.relu(mask, inplace=True)
        mask = self.conv2(mask)
        mask = self.sigmoid(mask)      # (B*, C, H, W) in (0,1)

        # x * (1 + mask): nếu mask lớn → khuếch đại mạnh hơn
        return x * (1.0 + mask)


# =========================
# 2. Motion CNN + Motion Irregularity Extractor (MIE)
# =========================
class MotionCNN(nn.Module):
    """
    CNN cho nhánh chuyển động (nhận input là multi-level temporal difference).
    Input:  (B*T, C_in, H, W)
    Output: (B*T, C_out, H', W')
    """
    def __init__(self, in_channels=6, base_channels=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
        )
        self.out_channels = base_channels * 2

    def forward(self, x):
        return self.conv(x)


class MotionIrregularityExtractor(nn.Module):
    """
    MIE: học đặc trưng chuyển động bất thường (flicker, jitter, warp sai).
    Input:  (B*, C, H, W)
    Output: (B*, C, H, W)
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        return out + x   # residual: giữ cả motion gốc + motion irregularity


# =========================
# 3. Cross Attention Artifact Fusion Block (CAFB)
# =========================
class CAFB(nn.Module):
    """
    Cross-Attention CNN-based giữa spatial artifact feature và motion feature.

    Input:
        A_s: (B, T, C, H, W)  - spatial artifact features
        M_t: (B, T, C, H, W)  - motion features

    Output:
        F_ca: (B, T, C, H, W) - motion được "hướng dẫn" bởi artifact
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        d = max(channels // reduction, 8)   # chiều ẩn cho Q,K

        self.conv_q = nn.Conv2d(channels, d, kernel_size=1)
        self.conv_k = nn.Conv2d(channels, d, kernel_size=1)
        self.conv_v = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        self.scale = d ** 0.5

    def forward(self, A_s, M_t):
        B, T, C, H, W = A_s.shape

        # (B*T, C, H, W)
        A_flat = A_s.view(B * T, C, H, W)
        M_flat = M_t.view(B * T, C, H, W)

        Q = self.conv_q(A_flat)   # (B*T, d, H, W)
        K = self.conv_k(M_flat)   # (B*T, d, H, W)
        V = self.conv_v(M_flat)   # (B*T, C, H, W)

        BT, d, H, W = Q.shape
        N = H * W

        Q = Q.view(BT, d, N)          # (BT, d, N)
        K = K.view(BT, d, N)          # (BT, d, N)
        V = V.view(BT, C, N)          # (BT, C, N)

        # attention scores trên từng vị trí H*W (dot-product theo channel d)
        att_scores = (Q * K).sum(dim=1) / self.scale   # (BT, N)
        att_map = F.softmax(att_scores, dim=-1)        # (BT, N)

        # apply attention lên V
        att_map = att_map.unsqueeze(1)                 # (BT, 1, N)
        F_ca = V * att_map                             # (BT, C, N)

        F_ca = F_ca.view(B, T, C, H, W)                # (B, T, C, H, W)
        return F_ca


# =========================
# 4. CAAF-Net (phiên bản cho input B, C, T, H, W)
# =========================
class CAAFNet(nn.Module):
    """
    CAAF-Net (CNN-based, 2-stream, cross-attention, bidirectional refinement)

    Input:
        x: (B, C, T, H, W)  - video frames (RGB)
    Output:
        logits: (B, num_classes)
    """
    def __init__(
        self,
        num_classes=2,
        use_lstm=True,
        lstm_hidden=256,
        lstm_layers=1,
        in_channels=3,
        use_efficientnet=True,
    ):
        super().__init__()

        # ---- Spatial branch backbone: EfficientNet hoặc Dummy CNN
        if use_efficientnet:
            self.spatial_stem = EfficientNetBackbone(model_name="efficientnet-b0",
                                                     pretrained=False,
                                                     in_channels=in_channels)
            spatial_out_channels = self.spatial_stem.out_channels
        else:
            # fallback nhỏ (nếu bạn chưa muốn dùng EfficientNet)
            self.spatial_stem = nn.Sequential(
                nn.Conv2d(in_channels, 32, 3, padding=1, stride=2),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, padding=1, stride=2),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
            spatial_out_channels = 64

        # Artifact Amplification Block
        self.aab = ArtifactAmplificationBlock(spatial_out_channels)

        # ---- Motion branch (CNN trên multi-level temporal difference)
        # Input cho motion_cnn: 2*C (D1 + D2)
        self.motion_cnn = MotionCNN(in_channels=2 * in_channels, base_channels=32)
        motion_out_channels = self.motion_cnn.out_channels

        # Motion Irregularity Extractor
        self.mie = MotionIrregularityExtractor(motion_out_channels)

        # ---- Align channels giữa 2 nhánh (nếu khác nhau)
        if motion_out_channels != spatial_out_channels:
            self.align_motion = nn.Conv2d(motion_out_channels, spatial_out_channels, kernel_size=1)
            motion_out_channels = spatial_out_channels
        else:
            self.align_motion = nn.Identity()

        # ---- Cross Attention Artifact Fusion Block
        self.cafb = CAFB(channels=spatial_out_channels)

        # ---- Bidirectional refinement convs
        self.refine_from_ca = nn.Conv2d(spatial_out_channels, spatial_out_channels, kernel_size=3, padding=1)
        self.refine_from_spatial = nn.Conv2d(spatial_out_channels, spatial_out_channels, kernel_size=3, padding=1)

        # ---- Temporal modeling (LSTM trên vector pooled)
        self.use_lstm = use_lstm
        feat_dim = spatial_out_channels * 2  # concat spatial + motion

        if use_lstm:
            self.lstm = nn.LSTM(
                input_size=feat_dim,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                batch_first=True,
                bidirectional=True,
            )
            cls_in_dim = lstm_hidden * 2
        else:
            # nếu không dùng LSTM thì mean pooling theo T
            cls_in_dim = feat_dim

        self.classifier = nn.Sequential(
            nn.Linear(cls_in_dim, cls_in_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(cls_in_dim // 2, num_classes),
        )

    def _build_temporal_diffs(self, x):
        """
        Multi-level temporal difference (kiểu TDN):

        x: (B, C, T, H, W)
        Trả về: motion (B, 2C, T, H, W)
            - D1_t = |X_t - X_{t-1}|
            - D2_t = |X_{t+1} - X_{t-1}|
        """
        B, C, T, H, W = x.shape

        # D1: first-order difference
        x_prev = torch.zeros_like(x)
        x_prev[:, :, 1:] = x[:, :, :-1]
        D1 = torch.abs(x - x_prev)    # (B, C, T, H, W)

        # D2: second-order / central difference
        x_prev2 = torch.zeros_like(x)
        x_next2 = torch.zeros_like(x)
        x_prev2[:, :, 1:] = x[:, :, :-1]
        x_next2[:, :, :-1] = x[:, :, 1:]
        D2 = torch.abs(x_next2 - x_prev2)   # (B, C, T, H, W)

        # concat theo channel: (B, 2C, T, H, W)
        motion = torch.cat([D1, D2], dim=1)
        return motion

    def forward(self, x):
        """
        x: (B, C, T, H, W)
        """
        B, C, T, H, W = x.shape

        # ====================
        # Spatial Artifact Branch
        # ====================
        # đổi sang (B*T, C, H, W) để cho qua backbone
        x_bt = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        spatial_feat = self.spatial_stem(x_bt)      # (B*T, Cs, Hs, Ws)
        Cs, Hs, Ws = spatial_feat.shape[1:]

        # AAB: khuếch đại artifact
        spatial_feat = self.aab(spatial_feat)       # (B*T, Cs, Hs, Ws)

        # reshape về (B, T, Cs, Hs, Ws)
        spatial_feat = spatial_feat.view(B, T, Cs, Hs, Ws)  # A_s

        # ====================
        # Motion Branch (multi-level temporal difference)
        # ====================
        motion = self._build_temporal_diffs(x)      # (B, 2C, T, H, W)
        motion_bt = motion.permute(0, 2, 1, 3, 4).contiguous().view(B * T, 2 * C, H, W)
        motion_feat = self.motion_cnn(motion_bt)    # (B*T, Cm, Hm, Wm)
        motion_feat = self.mie(motion_feat)         # (B*T, Cm, Hm, Wm)
        motion_feat = self.align_motion(motion_feat)  # (B*T, Cs, Hm, Wm)

        Cm, Hm, Wm = motion_feat.shape[1:]

        # nếu spatial và motion khác spatial size → resize
        if (Hm != Hs) or (Wm != Ws):
            motion_feat = F.interpolate(motion_feat, size=(Hs, Ws),
                                        mode="bilinear", align_corners=False)

        motion_feat = motion_feat.view(B, T, Cs, Hs, Ws)    # M_t

        # ====================
        # Cross Attention Artifact Fusion (CAFB)
        # ====================
        F_ca = self.cafb(spatial_feat, motion_feat)  # (B, T, Cs, Hs, Ws)

        # ====================
        # Bidirectional Refinement (BRL)
        # ====================
        # spatial được refine bởi F_ca
        spatial_ref = spatial_feat + self.refine_from_ca(
            F_ca.view(B * T, Cs, Hs, Ws)
        ).view(B, T, Cs, Hs, Ws)

        # motion được refine bởi spatial
        motion_ref = motion_feat + self.refine_from_spatial(
            spatial_feat.view(B * T, Cs, Hs, Ws)
        ).view(B, T, Cs, Hs, Ws)

        # ====================
        # Pooling + Temporal modeling
        # ====================
        # GAP trên H,W
        spatial_vec = spatial_ref.mean(dim=[3, 4])  # (B, T, Cs)
        motion_vec  = motion_ref.mean(dim=[3, 4])   # (B, T, Cs)

        # concat features của 2 branch
        feat = torch.cat([spatial_vec, motion_vec], dim=-1)  # (B, T, 2*Cs)

        if self.use_lstm:
            # LSTM theo T
            out, _ = self.lstm(feat)  # (B, T, 2*hidden)
            # lấy mean theo T (cũng có thể lấy step cuối)
            out = out.mean(dim=1)     # (B, 2*hidden)
        else:
            # mean theo T
            out = feat.mean(dim=1)    # (B, 2*Cs)

        logits = self.classifier(out)  # (B, num_classes)
        return logits
