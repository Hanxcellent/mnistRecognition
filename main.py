import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os

# 1. 定义超参数
batch_size = 64
learning_rate = 0.001
num_epochs = 10
model_save_path = './mnist_cnn.pth'  # 模型保存路径
output_dir = './output'  # 输出目录

# 创建输出目录（如果不存在的话）
os.makedirs(output_dir, exist_ok=True)

# 2. 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 均值和标准差
])

# 加载数据集（不下载）
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=False)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=False)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 3. 定义CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # 展平
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 4. 定义设备（GPU 如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 记录训练过程
train_losses = []
test_losses = []
test_accuracies = []

# 5. 训练模型
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch'):
        images, labels = images.to(device), labels.to(device)  # 将数据移动到GPU
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}')

    # 测试模型
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing', unit='batch'):
            images, labels = images.to(device), labels.to(device)  # 将数据移动到GPU
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss /= len(test_loader)
    test_losses.append(test_loss)
    accuracy = 100 * correct / total
    test_accuracies.append(accuracy)

    print(f'Epoch [{epoch+1}/{num_epochs}], Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')

# 6. 保存模型
torch.save(model.state_dict(), model_save_path)
print(f'Model saved to {model_save_path}')

# 7. 结果可视化 - 保存损失和准确率曲线
plt.figure(figsize=(12, 5))

# 绘制训练和测试损失
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(output_dir, 'loss_curve.png'))  # 保存为PNG
plt.close()

# 绘制测试准确率
plt.subplot(1, 2, 2)
plt.plot(test_accuracies, label='Test Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.savefig(os.path.join(output_dir, 'accuracy_curve.png'))  # 保存为PNG
plt.close()

# 8. 生成对照图 - 保存输入图片和预测结果
model.eval()
num_images = 10
images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)

with torch.no_grad():
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)

# 将张量转化为numpy数组用于绘图
images = images.cpu().numpy()
predicted = predicted.cpu().numpy()
labels = labels.cpu().numpy()

# 保存输入图片与预测结果对照的图
plt.figure(figsize=(12, 4))
for i in range(num_images):
    plt.subplot(2, num_images, i + 1)
    plt.imshow(images[i][0], cmap='gray')
    plt.title(f'True: {labels[i]}')
    plt.axis('off')

    plt.subplot(2, num_images, i + 1 + num_images)
    plt.imshow(images[i][0], cmap='gray')
    plt.title(f'Pred: {predicted[i]}')
    plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'predictions_comparison.png'))  # 保存为PNG
plt.close()
