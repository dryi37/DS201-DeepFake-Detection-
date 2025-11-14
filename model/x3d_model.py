import torch
import torch.nn as nn
from pytorchvideo.models.hub import x3d_l

class x3d_model(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        self.backbone = x3d_l(pretrained=pretrained)

        self.backbone.blocks[-1].pool.pool = nn.AdaptiveAvgPool3d(1)

        # Thay FC cuối thành head 2 lớp
        in_features = self.backbone.blocks[-1].proj.in_features
        self.backbone.blocks[-1].proj = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T, H, W)
        return: (B, num_classes)
        """
        return self.backbone(x)
    
if __name__ == "__main__":
    model = x3d_model(num_classes=2, pretrained=False)
    x = torch.randn(2, 3, 16, 224, 224)  # batch 2 clip
    out = model(x)
    print(out.shape)  # expected: [2, 2]
    print(f"Tổng số tham số: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")