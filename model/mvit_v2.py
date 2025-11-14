import torch
import torch.nn as nn
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights

class MViT_v2_S(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, dropout=0.2):
        super().__init__()
        if pretrained:
            weights = MViT_V2_S_Weights.KINETICS400_V1
            self.backbone = mvit_v2_s(weights=weights)
        else:
            self.backbone = mvit_v2_s(weights=None)

        in_features = self.backbone.head[1].in_features
        self.backbone.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


if __name__ == "__main__":
    model = MViT_v2_S(num_classes=2, pretrained=True)
    model.eval()

    dummy = torch.randn(2, 3, 16, 224, 224)
    out = model(dummy)
    print("Output shape:", out.shape)