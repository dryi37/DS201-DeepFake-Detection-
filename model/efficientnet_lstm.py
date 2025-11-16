import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class efficientnet_lstm(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True, hidden_size: int = 256, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()

        self.backbone = EfficientNet.from_pretrained('efficientnet-b0') if pretrained else EfficientNet.from_name('efficientnet-b0')
        
        # Thay FC cuối thành head LSTM + 2 lớp
        in_features = self.backbone._fc.in_features
        self.backbone._fc = nn.Identity()  # Loại bỏ lớp fully connected cuối cùng

        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # self.attn = nn.Linear(hidden_size * 2, 1)

        direction = 2
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * direction, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

        self._init_weights()

        if not pretrained:
            self._init_backbone(self.backbone)

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

        # nn.init.xavier_uniform_(self.attn.weight)
        # nn.init.constant_(self.attn.bias, 0)

        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _init_backbone(self, module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        B, C, T, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(B * T, C, H, W)

        feat = self.backbone(x)  # (B*T, in_features)
        feat = feat.view(B, T, -1)  # (B, T, in_features)

        lstm_out, _ = self.lstm(feat)  # (B, T, hidden_size*direction)
        
        # Mean pooling
        out = torch.mean(lstm_out, dim=1)  # (B, hidden_size*direction)
        out = self.fc(out)  # (B, num_classes)
        
        # Attention pooling
        # weights = torch.softmax(self.attn(lstm_out), dim=1)
        # pooled = (weights * lstm_out).sum(dim=1)
        # out = self.fc(pooled)
        return out