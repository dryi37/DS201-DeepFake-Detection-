import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class efficientnet_lstm_finetune(nn.Module):
    def __init__(self, base_model, num_classes=2, dropout=0.2):
        super().__init__()

        # 1. Load backbone + LSTM + FC từ model cũ
        self.backbone = base_model.backbone
        self.lstm = base_model.lstm

        hidden_dim = self.lstm.hidden_size * 2

        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.lstm.parameters():
            p.requires_grad = False

        self.frame_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        B, C, T, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4)     # (B,T,C,H,W)
        x = x.reshape(B * T, C, H, W)

        feat = self.backbone(x)          # (B*T, D)
        feat = feat.view(B, T, -1)       # (B, T, D)

        lstm_out, _ = self.lstm(feat)    # (B, T, H*2)

        frame_logits = self.frame_head(lstm_out)  # (B, T, num_classes)

        video_logits = torch.mean(frame_logits, dim=1)  # (B, num_classes)

        return video_logits, frame_logits
