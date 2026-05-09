import os
import torch
from torch.utils.data import DataLoader, random_split
from dataset import QuickDrawDataset
from model import SketchCNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

dataset = QuickDrawDataset(
    data_dir=os.path.join(BASE_DIR, "data"),
    categories_file=os.path.join(BASE_DIR, "data", "categories.txt"),
    max_per_class=15000
)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=128)

model = SketchCNN(num_classes=len(dataset.classes)).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', patience=2, factor=0.5
)

criterion = torch.nn.CrossEntropyLoss()

best_acc = 0

def evaluate():
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total


for epoch in range(25):
    model.train()
    total_loss = 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        preds = model(x)
        loss = criterion(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    acc = evaluate()
    scheduler.step(acc)

    print(f"Epoch {epoch} | loss={total_loss:.2f} | val_acc={acc:.3f}")

    # 💾 SAVE BEST MODEL
    if acc > best_acc:
        best_acc = acc
        os.makedirs("../weights", exist_ok=True)

        torch.save({
            "model_state": model.state_dict(),
            "classes": dataset.classes
        }, "../weights/best_model.pt")

        print("🔥 model saved")

print("✅ TRAINING DONE. BEST ACC:", best_acc)