import torch
from torch import nn


import torch
from torch import nn
from transformers import ViTModel


class Model(nn.Module):
    """
    Vision Transformer classifier wrapper.
    Backbone: Hugging Face ViT
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        # Load pretrained backbone
        self.backbone = ViTModel.from_pretrained(model_name)

        # Optionally freeze backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        hidden_size = self.backbone.config.hidden_size

        # Simple classification head
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=x)

        # CLS token representation
        cls_embedding = outputs.last_hidden_state[:, 0]

        return self.classifier(cls_embedding)



if __name__ == "__main__":
    model = Model()
    x = torch.rand(1)
    print(f"Output shape of model: {model(x).shape}")
