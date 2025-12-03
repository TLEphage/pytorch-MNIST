# pytorch-MNIST
PyTorch 入门实战：基于 MNIST 手写数字数据集的神经网络训练与测试

## 项目介绍
这是一个面向机器学习新手的入门项目，核心目标是：
1. 熟悉 PyTorch 框架的基本使用（数据加载、模型定义、训练循环、参数优化）
2. 理解简单神经网络的工作原理（输入层、隐藏层、输出层）
3. 掌握 MNIST 数据集的处理流程（经典手写数字识别任务）
4. 完成端到端的模型训练、测试与效果验证

MNIST 数据集包含 60000 张训练图和 10000 张测试图，每张图是 28×28 像素的灰度手写数字（0-9），任务是训练模型准确识别数字类别。

## 环境准备
### 1. 依赖库列表
- Python 3.8+（推荐 3.9/3.10）
- PyTorch 1.10+（核心框架）
- torchvision 0.11+（数据集加载+预处理）
- numpy 1.21+（安装 pytorch 时自动安装，数据处理辅助）
- matplotlib 3.5+（可选，可视化训练过程）

### 2. 安装依赖

安装 miniconda 并创建虚拟环境

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
conda create -n pytorch-312-cuda128 python=3.12
conda env list
conda activate pytorch-312-cuda128
```

安装 Pytorch 和 matplotlib 

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.8 -c pytorch -c nvidia

conda install matplotlib
```

### 3. 验证环境

```bash
python -c "import torch; print(f'PyTorch版本：{torch.__version__}'); print(f'CUDA是否可用：{torch.cuda.is_available()}')"

python -c "import numpy; print(f'numpy版本：{numpy.__version__}')"

python -c "import torchvision ; print(f'torchvision 版本：{torchvision .__version__}')"

python -c "import matplotlib; print(f'matplotlib版本：{matplotlib.__version__}')"
```

## 快速开始

### 1. 克隆仓库
```bash
git clone https://github.com/TLEphage/pytorch-MNIST.git
cd pytorch-MNIST
```

### 2. 运行训练脚本
```bash
python test.py
```

### 3. 查看结果

- 训练过程中会打印每轮（epoch）和测试准确率（accuracy）
- 训练结束后，会展示三份测试集的图片和对应预测结果

## 项目结构

```plaintext
pytorch-MNIST/
├── README.md          # 项目说明（本文档）
└── test.py            # 入门级训练代码
```

## 参考资料
- [PyTorch 官方文档](https://pytorch.org/docs/stable/index.html)
- [MNIST 数据集官网](http://yann.lecun.com/exdb/mnist/)
- [PyTorch 入门教程（菜鸟教程）](https://www.runoob.com/pytorch/pytorch-tutorial.html)
