"""
本代码为最基础的MNIST训练代码，使用全连接神经网络（FCN）训练
参考视频: https://www.bilibili.com/video/BV1GC4y15736/?spm_id_from=333.337.search-card.all.click&vd_source=7156d8982eb4ed33bd3b29ca9f7322d9
"""

import torch 
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

class Net(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28*28, 64) # 通过矩阵乘法讲 28*28 维张量变为 64 维张量
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.log_softmax(self.fc4(x)) # 先使所有维之和为1，再分别取ln，防止大数值溢出
        return x
    
def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()]) # 把数据集的图像转换成 PyTorch 能处理的张量（Tensor）
    # (28, 28, val=0~255) -> (1, 28, 28, val=0~1.0) , 1 表示1个通道 
    data_set = MNIST("", is_train, transform=to_tensor, download=True)
    return DataLoader(data_set, batch_size=15, shuffle=True)

def evaluate(test_data, net):
    n_correct = 0
    n_total = 0
    with torch.no_grad(): # 禁用梯度计算（评估时无需反向传播）
        for (x,y) in test_data:
            outputs = net.forward(x.view(-1, 28*28))
            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]:
                    n_correct += 1
                n_total += 1
    return n_correct / n_total

def main():
    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    net = Net()

    print("initial accuracy:", evaluate(test_data, net))

    # 定义优化器（负责更新模型参数）
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(5):
        for (x,y) in train_data:
            net.zero_grad()
            output = net.forward(x.view(-1, 28*28)) # x.view(-1, 28*28) 表示把 28*28矩阵转为 784维张量，-1表示自动计算第一维大小
            loss = torch.nn.functional.nll_loss(output, y)  # (15, 10) 张量
            loss.backward() # 计算梯度
            optimizer.step() # 根据上一步结果更新模型参数
        print("epoch", epoch, "accuracy: ", evaluate(test_data, net))

    # 可视化前4个测试样本的预测结果
    for (n, (x, _)) in enumerate(test_data):
        if n > 3:
            break
        predict = torch.argmax(net.forward(x[0].view(-1, 28*28)))
        plt.figure(n)
        plt.imshow(x[0].view(28, 28))
        plt.title("prediction: " + str(int(predict)))
    plt.show()

if __name__ == "__main__":
    main()