import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# ─── 超参数 ───────────────────────────────────────────
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
DATA_DIR = "./data"
MODEL_PATH = "./mnist_model.pth"

# ─── 数据集 ───────────────────────────────────────────
# 将图像像素值从 [0,255] 归一化到 [-1,1]，有助于训练稳定
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 的均值和标准差
])

train_dataset = datasets.MNIST(root=DATA_DIR, train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print(f"训练集大小: {len(train_dataset)} 张图像")
print(f"测试集大小: {len(test_dataset)} 张图像")

# ─── 神经网络模型 ─────────────────────────────────────
# 结构: 784 → 256 → 128 → 10
# ReLU 是激活函数，BatchNorm 帮助训练更稳定，Dropout 防止过拟合
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),               # 28×28 → 784
            nn.Linear(784, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)          # 10 个类别，不加 Softmax（CrossEntropyLoss 内含）
        )

    def forward(self, x):
        return self.model(x)

device = torch.device("cpu")
model  = MLP().to(device)
print(f"\n模型结构:\n{model}")
print(f"总参数量: {sum(p.numel() for p in model.parameters()):,}")

# ─── 损失函数 & 优化器 ────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
# 学习率调度：每 3 个 epoch 将学习率乘以 0.5
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

# ─── 训练函数 ─────────────────────────────────────────
def train_one_epoch(epoch):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 200 == 0:
            print(f"  Epoch {epoch} [{batch_idx+1}/{len(train_loader)}]  "
                  f"Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

# ─── 评估函数 ─────────────────────────────────────────
def evaluate():
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return 100. * correct / total

# ─── 主训练循环 ───────────────────────────────────────
train_losses, train_accs, test_accs = [], [], []

print("\n开始训练...\n" + "="*50)
for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(epoch)
    test_acc = evaluate()
    scheduler.step()

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

    print(f"Epoch {epoch:2d}/{EPOCHS}  |  "
          f"Train Loss: {train_loss:.4f}  |  "
          f"Train Acc: {train_acc:.2f}%  |  "
          f"Test Acc: {test_acc:.2f}%")

print("="*50)
print(f"\n最终测试准确率: {test_accs[-1]:.2f}%")
if test_accs[-1] >= 95:
    print("✓ 达到目标准确率 (≥95%)")
else:
    print("✗ 未达到目标，可尝试增加 EPOCHS")

# ─── 保存模型 ─────────────────────────────────────────
torch.save(model.state_dict(), MODEL_PATH)
print(f"\n模型已保存至: {MODEL_PATH}")

# ─── 绘制训练曲线 ─────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(range(1, EPOCHS+1), train_losses, marker='o')
ax1.set_title("Training Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.grid(True)

ax2.plot(range(1, EPOCHS+1), train_accs, marker='o', label="Train")
ax2.plot(range(1, EPOCHS+1), test_accs,  marker='s', label="Test")
ax2.axhline(y=95, color='r', linestyle='--', label="Target 95%")
ax2.set_title("Accuracy")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("./training_curve.png", dpi=150)
print("训练曲线已保存至: training_curve.png")
plt.show()
