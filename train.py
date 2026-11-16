
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import MyDataset, split_dataset
from model import SimpleTCN


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # 1. Загружаем датасет
    dataset = MyDataset(
        stride=5,
        window_frames=30,
        window_seconds=3
    )

    # 2. Разделяем по исходным видео
    train_dataset, val_dataset = split_dataset(
        dataset,
        val_ratio=0.2
    )

    # 3. Создаём загрузчики батчей
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False
    )

    # 4. Создаём модель
    model = SimpleTCN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    # 5. Обучение
    for epoch in range(20):
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(X)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y)
            train_correct += (
                logits.argmax(dim=1) == y
            ).sum().item()
            train_total += len(y)

        # 6. Валидация
        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                y = y.to(device)

                logits = model(X)
                loss = criterion(logits, y)

                val_loss += loss.item() * len(y)
                val_correct += (
                    logits.argmax(dim=1) == y
                ).sum().item()
                val_total += len(y)

        print(
            f"Epoch {epoch + 1:02d} | "
            f"Train loss: {train_loss / train_total:.4f} | "
            f"Train acc: {train_correct / train_total:.2%} | "
            f"Val loss: {val_loss / val_total:.4f} | "
            f"Val acc: {val_correct / val_total:.2%}"
        )

    # 7. Сохраняем веса
    from pathlib import Path
    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "models/fall_tcn.pt")


if __name__ == "__main__":
    main()
