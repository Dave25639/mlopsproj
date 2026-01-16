# src/mlopsproj/train.py

import torch
from torch.utils.data import DataLoader

from mlopsproj.data import MyDataset
from mlopsproj.model import Model


def train():
    # --- Dataset ---
    dataset = MyDataset("data/raw")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # --- Model ---
    model = Model(
        name="google/vit-base-patch16-224-in21k",
        num_classes=2,          # change when dataset is real
        freeze_backbone=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    # --- Optimizer ---
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    # --- Training loop (minimal) ---
    for epoch in range(1):
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")


if __name__ == "__main__":
    train()
