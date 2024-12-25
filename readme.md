# MNIST手写数字识别（CNN）

这个项目使用PyTorch实现卷积神经网络（CNN）来识别MNIST数据集中的手写数字。模型经过训练和评估，并可视化预测结果与实际标签的比较。

## 环境要求

要运行此代码，需要安装所需的Python包。你可以使用以下命令通过pip安装：

```bash
pip install -r requirements.txt
```
requirements.txt中包含以下内容：

- PyTorch（根据适当的CUDA版本）
- torchvision
- tqdm
- matplotlib

## 数据集
MNIST数据集包含28x28像素的灰度图像，表示手写数字（0-9）。你可以手动下载该数据集，并将其放在./data目录中。代码将从此目录加载数据集。

## 使用方法
要运行训练过程，只需执行主脚本。模型将进行训练，然后将其权重保存到mnist_cnn.pth。在训练过程中，将绘制损失和准确率曲线，并将其保存为PNG文件到output目录中，同时保存输入图像与预测结果的比较图。
````
python train.py
````

### 输出结果
- 训练好的模型将保存到mnist_cnn.pth。
- 损失和准确率曲线将保存到output目录中：
  - loss_curve.png
  - accuracy_curve.png
- 输入图像与其预测结果的比较：
  - predictions_comparison.png