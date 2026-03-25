import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms, models
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import os

class CIFAR10_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # 第一组卷积块
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),

            # 第二组卷积块
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.4),

            # 第三组卷积块
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.5),

            # 全局平均池化替代 Flatten + Linear (减少参数，防止过拟合)
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),

            # 全连接层
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        return self.model(x)


# CIFAR-10 数据集的标准化参数
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2023, 0.1994, 0.2010]

# 训练集：数据增强 + 归一化
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
    transforms.RandomCrop(32, padding=4),  # 随机裁剪
    transforms.ColorJitter(brightness=0.2,  # 颜色抖动
                           contrast=0.2,
                           saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
])

# 测试集：仅归一化
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
])

# 加载数据集
train_data = torchvision.datasets.CIFAR10(
    root='./dataset',
    train=True,
    download=True,
    transform=train_transform
)

test_data = torchvision.datasets.CIFAR10(
    root='./dataset',
    train=False,
    download=True,
    transform=test_transform
)

train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练集大小：{train_data_size}, 测试集大小：{test_data_size}")

# 数据加载器
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

#恢复训练配置
RESUME_TRAINING = True  # 设置为 True 则恢复训练
CHECKPOINT_PATH = "checkpoints/checkpoint_epoch80.pth"  # 指定检查点路径


# 创建网络模型
net = CIFAR10_Net()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net.to(device)
print(f"使用设备：{device}")

# 损失函数
criterion = nn.CrossEntropyLoss()

# 优化器
learning_rate = 0.1
optimizer = optim.SGD(
    net.parameters(),
    lr=learning_rate,
    momentum=0.9,
    weight_decay=5e-4  # L2 正则化
)

# 学习率调度器：余弦退火
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

# 训练参数设置
total_train_step = 0
total_test_step = 0
epoch = 100
best_accuracy = 0.0
start_epoch = 0

if RESUME_TRAINING and os.path.exists(CHECKPOINT_PATH):
    print(f"加载检查点：{CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device,weights_only=True)

    # 加载模型状态
    net.load_state_dict(checkpoint['model_state_dict'])

    # 加载优化器状态
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 恢复 epoch
    start_epoch = checkpoint['epoch']

    # 恢复最佳准确率
    best_accuracy = checkpoint.get('accuracy', 0.0)

    # 恢复学习率调度器状态
    scheduler.load_state_dict(checkpoint.get('scheduler_state_dict', {}))

    # 恢复 step 计数
    total_test_step = start_epoch
    total_train_step = start_epoch * len(train_loader)

    print(f"成功恢复训练！从第 {start_epoch + 1} 轮开始")
    print(f"当前最佳准确率：{best_accuracy:.4f}")
else:
    print("从头开始训练")

# TensorBoard 可视化
writer = SummaryWriter("logs/CIFAR10_Optimized")

# 创建模型保存目录
os.makedirs("checkpoints", exist_ok=True)


for i in range(start_epoch,epoch):
    print(f"\n{'=' * 50}")
    print(f"第 {i + 1}/{epoch} 轮训练开始")
    print(f"当前学习率：{optimizer.param_groups[0]['lr']:.6f}")
    print(f"{'=' * 50}")

    #训练阶段
    net.train()
    total_train_loss = 0.0
    train_correct = 0

    for batch_idx, (imgs, targets) in enumerate(train_loader):
        imgs = imgs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = net(imgs)
        loss = criterion(outputs, targets)
        loss.backward()

        # 梯度裁剪 (防止梯度爆炸)
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

        optimizer.step()

        total_train_loss += loss.item()
        train_correct += (outputs.argmax(1) == targets).sum().item()
        total_train_step += 1

        # 每 100 步记录一次训练 loss
        if total_train_step % 100 == 0:
            writer.add_scalar("train_loss_step", loss.item(), total_train_step)

    # 更新学习率
    scheduler.step()

    # 计算训练集准确率
    train_accuracy = train_correct / train_data_size
    avg_train_loss = total_train_loss / len(train_loader)

    print(f"训练集 Loss: {avg_train_loss:.4f}")
    print(f"训练集准确率：{train_accuracy:.4f}")

    #测试阶段
    net.eval()
    total_test_loss = 0.0
    test_correct = 0

    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = net(imgs)

            loss = criterion(outputs, targets)
            total_test_loss += loss.item()
            test_correct += (outputs.argmax(1) == targets).sum().item()

    test_accuracy = test_correct / test_data_size
    avg_test_loss = total_test_loss / len(test_loader)

    print(f"测试集 Loss: {avg_test_loss:.4f}")
    print(f"测试集准确率：{test_accuracy:.4f}")

    # 记录到 TensorBoard
    writer.add_scalars("Loss", {
        'train': avg_train_loss,
        'test': avg_test_loss
    }, total_test_step)

    writer.add_scalars("Accuracy", {
        'train': train_accuracy,
        'test': test_accuracy
    }, total_test_step)

    writer.add_scalar("Learning_Rate", optimizer.param_groups[0]['lr'], total_test_step)

    total_test_step += 1

    # 保存最佳模型
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        torch.save({
            'epoch': i + 1,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'accuracy': test_accuracy,
        }, f"checkpoints/best_model_epoch{i + 1}_acc{test_accuracy:.4f}.pth")
        print(f" 保存最佳模型,准确率：{test_accuracy:.4f}")

    # 每 10 轮保存一次检查点
    if (i + 1) % 10 == 0:
        torch.save({
            'epoch': i + 1,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': test_accuracy,
        }, f"checkpoints/checkpoint_epoch{i + 1}.pth")

print(f"训练完成！")
print(f"最佳测试准确率：{best_accuracy:.4f}")

writer.close()
