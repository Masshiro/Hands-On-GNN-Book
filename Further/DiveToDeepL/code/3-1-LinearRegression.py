import random
import torch
import numpy as np
from torch.utils import data
from torch import nn

"""
1. 从零实现
"""
def synthetic_data(w, b, nums_examples):
    """生成形如 y = Xw + b + noise 的数据集"""
    X = torch.normal(0, 1, (nums_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

def data_iter(batch_size, features, labels):
    """读取数据集"""
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices) # 随机读取
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

def linreg(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):
    """均⽅损失""" 
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():           # 上下文管理器来禁用自动求导功能，因为我们不需要计算参数的二阶导数
        for param in params:
            # 参数更新方法：
            # 用参数当前的值减去学习率乘以参数当前的梯度除以小批量数据的大小
            param -= lr * param.grad / batch_size
            param.grad.zero_()

def main_scratch():
    # 定义超参数
    lr = 0.03
    num_epochs = 5
    net = linreg
    loss = squared_loss

    # 生成数据集
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)

    # 初始化模型参数
    batch_size = 10
    w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    # 训练
    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y) # X和y的小批量损失
            # 因为l形状是(batch_size,1)，⽽不是⼀个标量。l中的所有元素被加到⼀起， 
            # 并以此计算关于[w,b]的梯度
            l.sum().backward()
            sgd([w, b], lr, batch_size) # 使用参数的梯度更新参数
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
    
    # 显示误差
    print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}') 
    print(f'b的估计误差: {true_b - b}')

"""
2. 简洁实现
"""
def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

def main_concise():
    # 定义超参数
    lr = 0.03
    num_epochs = 5

    # 生成数据集
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)

    # 读取数据集
    batch_size = 10
    data_iter = load_array((features, labels), batch_size)

    # 定义模型
    net = nn.Sequential(nn.Linear(2, 1))

    # 初始化模型参数
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)

    # 定义损失函数
    loss = nn.MSELoss()

    # 定义优化函数
    trainer = torch.optim.SGD(net.parameters(), lr)

    # 训练
    for epoch in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y)     # ⽣成预测并计算损失（前向传播）
            trainer.zero_grad()
            l.backward()            # 反向传播计算梯度
            trainer.step()          # 使用优化器更新参数
        l = loss(net(features), labels)
        print(f'epoch {epoch + 1}, loss {l:f}')
    
    # 显示误差
    w = net[0].weight.data 
    print('w的估计误差：', true_w - w.reshape(true_w.shape)) 
    b = net[0].bias.data 
    print('b的估计误差：', true_b - b)

if __name__ == "__main__":
    print("从零实现：")
    main_scratch()
    print("\n简洁实现：")
    main_concise()