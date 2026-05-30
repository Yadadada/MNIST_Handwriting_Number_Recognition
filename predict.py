import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import sys
import os

# 修复中文字体
_zh_fonts = [f.name for f in fm.fontManager.ttflist if 'Microsoft YaHei' in f.name or 'SimHei' in f.name or 'SimSun' in f.name]
plt.rcParams['font.sans-serif'] = (_zh_fonts[:1] or ['DejaVu Sans']) + plt.rcParams['font.sans-serif']
plt.rcParams['axes.unicode_minus'] = False

MODEL_PATH = "./mnist_model.pth"

# ─── 模型定义（必须与 train.py 一致）────────────────────
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)

# ─── 加载模型 ─────────────────────────────────────────
def load_model():
    model = MLP()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

# ─── 图像预处理 ───────────────────────────────────────
# MNIST 是：白色数字 + 黑色背景，像素归一化到 [-1,1]
def preprocess(image_path):
    img = Image.open(image_path).convert("L")   # 转灰度

    # 自动判断背景色：如果背景是白色则反转（变成黑底白字）
    pixels = list(img.getdata())
    avg = sum(pixels) / len(pixels)
    if avg > 127:
        img = ImageOps.invert(img)

    img = img.resize((28, 28), Image.LANCZOS)   # 缩放到 28×28

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    tensor = transform(img).unsqueeze(0)        # 增加 batch 维度 → [1,1,28,28]
    return img, tensor

# ─── 推断 ────────────────────────────────────────────
def predict(image_path):
    if not os.path.exists(MODEL_PATH):
        print(f"错误：找不到模型文件 {MODEL_PATH}，请先运行 train.py")
        sys.exit(1)

    if not os.path.exists(image_path):
        print(f"错误：找不到图片文件 {image_path}")
        sys.exit(1)

    model = load_model()
    img, tensor = preprocess(image_path)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]  # 各类别概率
        predicted = probs.argmax().item()
        confidence = probs[predicted].item() * 100

    # ─── 可视化结果 ───────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # 左图：原始图像
    ax1.imshow(img, cmap="gray")
    ax1.set_title(f"输入图像", fontsize=13)
    ax1.axis("off")

    # 右图：各数字的概率柱状图
    bars = ax2.bar(range(10), probs.numpy() * 100, color="steelblue")
    bars[predicted].set_color("tomato")          # 高亮预测类别
    ax2.set_xticks(range(10))
    ax2.set_xlabel("数字类别")
    ax2.set_ylabel("概率 (%)")
    ax2.set_title(f"预测结果：{predicted}（置信度 {confidence:.1f}%）", fontsize=13)
    ax2.set_ylim(0, 105)
    ax2.grid(axis="y", alpha=0.3)

    for i, (bar, p) in enumerate(zip(bars, probs)):
        if p > 0.01:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f"{p*100:.1f}%", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out_path = "./prediction_result.png"
    plt.savefig(out_path, dpi=150)
    print(f"预测结果：数字 {predicted}，置信度 {confidence:.1f}%")
    print(f"结果图已保存至：{out_path}")
    plt.show()

# ─── 入口 ────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法：python predict.py <图片路径>")
        print("示例：python predict.py my_digit.png")
        sys.exit(1)

    predict(sys.argv[1])
