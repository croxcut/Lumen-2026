import torch
import torch.nn as nn
import timm


class LungMaxViTHybrid(nn.Module):
    def __init__(self, num_labels=6, dropout=0.3, pretrained=True):
        super(LungMaxViTHybrid, self).__init__()

        self.backbone = timm.create_model(
            'maxvit_tiny_tf_224',
            pretrained=pretrained,
            num_classes=0,
        )

        feature_dim = self.backbone.num_features

        # Freeze entire backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        print(f"Backbone frozen. Training classifier head only.")

        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(p=dropout / 4),
            nn.Linear(128, num_labels),
        )

    def forward(self, x):
        features = self.backbone(x)
        out      = self.classifier(features)
        return out


if __name__ == "__main__":
    model = LungMaxViTHybrid(num_labels=6)
    dummy = torch.randn(2, 3, 224, 224)
    out   = model(dummy)
    print(f"Output shape: {out.shape}")
    print("Model OK!")